# -*- coding: utf-8 -*-
"""
MiniGPT-4 (Vicuna-7B-v0) as the MSRC BASE MODEL (generator + self-verifier),
following the multi-model evaluation convention of VCD/OPERA/HALC. Built
against the MiniGPT-4 code vendored from the OPERA repo (minigpt4/), so the
generator is byte-identical to the one OPERA/HALC report CHAIR numbers on.

Duck-typed to the SAME interface as msrc.qwen_base.QwenBaseModel, so every
model-agnostic module (msrc.loop, msrc.tta, msrc.lattice, msrc.guards)
is reused VERBATIM:

  _prep_image(path)              -> returns the PATH (see note below)
  verify(q, img)                 -> p(Yes)/(p(Yes)+p(No))
  yes_no_gap(msg, img_or_None)   -> logsumexp(yes) - logsumexp(no)
  generate_sentence(img, prefix) -> (text, hit_eos)
  .tokenizer                     -> for token counting
  .VERIF_SUFFIXES                -> prompt-ensembling templates

WHY _prep_image RETURNS A PATH
------------------------------
Qwen-VL embeds the image path in the token stream, so msrc/tta.py's crop
TTA writes crops to disk and passes PATHS to the verifier. MiniGPT-4 instead
consumes a preprocessed TENSOR. To keep msrc/tta.py working verbatim (and
to keep the two lanes comparable), this adapter also uses the PATH as the
image handle and caches path -> Q-Former image embeddings internally. The
cache is sized to hold one image plus its 5 TTA crops, so a TTA'd image is
encoded once and reused across every claim on that image.

ARCHITECTURE NOTES (differences from the Qwen lane, all handled here)
--------------------------------------------------------------------
* MiniGPT-4 runs on inputs_embeds, not input_ids. Prompts are assembled as
  [BOS] + text-embeds + 32 image-embeds + text-embeds via get_context_emb.
* transformers-4.29.2 seeds generation with a [B,1] BOS token when
  inputs_embeds is used, so generated ids carry a leading BOS -> stripped.
* end_sym is "###": generation stops on a sentence end OR on "###".
* The text-only (language-prior) pass simply drops the <Img> block, giving
  a pure LLM prior with no visual tokens at all.
"""
import os
import re
import sys
import math
from argparse import Namespace

import types as _types
import importlib.abc as _ilib_abc
import importlib.util as _ilib_util

import torch
from torchvision import transforms
from transformers import StoppingCriteria, StoppingCriteriaList

# --- prompts -----------------------------------------------------------------
# Instruction template is OPERA's INSTRUCTION_TEMPLATE["minigpt4"].
PROMPT_TEMPLATE = "###Human: <Img><ImageHere></Img> {} ###Assistant:"
PROMPT_TEMPLATE_TEXTONLY = "###Human: {} ###Assistant:"

VERIF_SUFFIX = "\nAnswer the question using a single word or phrase."
# OPERA's chair_eval.py caption prompt, kept verbatim for comparability.
CAPTION_PROMPT = "Please describe this image in detail."

# BLIP-2 / EVA-CLIP normalization (OPERA sets do_normalize=False on the
# eval processor and applies this explicitly; we replicate that exactly).
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Training-only dependency stubs (see MiniGPT4BaseModel.__init__)
# ---------------------------------------------------------------------------
# Training-only dependency stubs
#
# minigpt4/__init__.py eagerly imports the FULL package (dataset builders,
# tasks, runners), dragging in deps MSRC inference never uses: decord (video;
# no py3.13 wheels), webdataset, skimage, matplotlib, scipy.
#
# The stub must survive what that code does AT IMPORT TIME:
#   from decord import VideoReader        -> attribute access
#   decord.bridge.set_bridge("torch")     -> module-level CALL
#   class ChainDataset(wds.DataPipeline)  -> module-level SUBCLASS
# So each stub attribute is a CLASS that no-ops: callable, subclassable and
# attribute-chainable. Real installs are never shadowed (find_spec is checked
# first; the finder is appended to the END of sys.meta_path). Any use is
# reported once so nothing fails silently.
# ---------------------------------------------------------------------------
_TRAINING_ONLY = ("decord", "webdataset", "braceexpand", "skimage",
                  "matplotlib", "scipy")
_STUB_USED = set()


def _stub_note(name):
    root = name.split(".")[0]
    if root not in _STUB_USED:
        _STUB_USED.add(root)
        print(f"[minigpt4-base] NOTE: stubbed '{root}' exercised at import "
              f"time ({name}) - expected for training-only paths.", flush=True)


def _make_stub(name):
    meta = type("_StubMeta", (type,), {
        "__getattr__": lambda cls, a: _make_stub(f"{name}.{a}"),
        "__repr__": lambda cls: f"<stub {name}>",
        "__iter__": lambda cls: iter(()),
    })
    return meta(name.split(".")[-1] or "stub", (object,), {
        "__init__": lambda self, *a, **k: _stub_note(name),
        "__getattr__": lambda self, a: _make_stub(f"{name}.{a}"),
        "__call__": lambda self, *a, **k: _make_stub(f"{name}()")(),
        "_stub_name": name,
    })


class _StubLoader(_ilib_abc.Loader):
    def create_module(self, spec):
        m = _types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = lambda a, _n=spec.name: _make_stub(f"{_n}.{a}")
        return m

    def exec_module(self, module):
        pass


class _StubFinder(_ilib_abc.MetaPathFinder):
    def __init__(self, names):
        self.names = tuple(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in self.names:
            return None
        return _ilib_util.spec_from_loader(fullname, _StubLoader())


def _stub_absent_training_deps(names=_TRAINING_ONLY):
    """Stub ONLY genuinely-absent modules; a real install always wins."""
    missing = []
    for n in names:
        if n in sys.modules:
            continue
        try:
            if _ilib_util.find_spec(n) is not None:
                continue
        except Exception:
            pass
        missing.append(n)
    if missing:
        sys.meta_path.append(_StubFinder(missing))
        print(f"[minigpt4-base] stubbed absent training-only deps: "
              f"{', '.join(missing)}", flush=True)
    return missing

_SENT_END = re.compile(r"[.!?]")


class MiniGPT4BaseModel:
    """MSRC base model backed by MiniGPT-4 (OPERA's implementation)."""

    # Suffix templates for prompt ensembling (index 0 = POPE-standard).
    VERIF_SUFFIXES = [
        VERIF_SUFFIX,
        "\nPlease answer yes or no.",
        "",
    ]

    def __init__(self, cfg_path="eval_configs/minigpt4_msrc_eval.yaml",
                 gpu_id=0, opera_root=None, cache_size=8, options=None):
        """
        cfg_path : MiniGPT-4 eval yaml (points at the local ckpt + vicuna).
        gpu_id   : index used for MiniGPT-4's device_8bit / low_resource path.
        opera_root: optional path to an OPERA checkout to import minigpt4
                    from, instead of the copy vendored next to this package.
        options  : list of "key=value" overrides merged into the config
                   (same syntax as OPERA's --options).
        """
        if opera_root:
            sys.path.insert(0, os.path.abspath(opera_root))

        # Import the packages for their REGISTRATION side effects, exactly
        # as OPERA's chair_eval.py does with `from minigpt4.models import *`.
        # A star import is illegal inside a function, and importing the
        # module is what actually populates the registry, so use importlib.
        import importlib
        # minigpt4/__init__.py eagerly imports the FULL package (dataset
        # builders, tasks, runners), which drags in training-only and
        # visualization-only deps: decord (video, no py3.13 wheels),
        # webdataset, skimage, matplotlib, scipy. None are reachable from
        # MSRC inference. Stub whichever are genuinely absent so the eager
        # import chain completes; real installs are never shadowed, and any
        # actual use raises loudly rather than failing silently.
        _stub_absent_training_deps()
        for _m in ("minigpt4.datasets.builders", "minigpt4.models",
                   "minigpt4.processors", "minigpt4.runners",
                   "minigpt4.tasks"):
            importlib.import_module(_m)

        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from minigpt4.models import load_preprocess

        args = Namespace(cfg_path=cfg_path, options=options or [],
                         gpu_id=gpu_id)
        cfg = Config(args)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device visible. Usually CUDA_VISIBLE_DEVICES points "
                "at a GPU index that does not exist (check GPU= in "
                "run_minigpt4_full.sh against `nvidia-smi`). Loading on CPU "
                "silently breaks fp16: 'self and mat2 must have the same "
                "dtype, but got Float and Half'.")
        self.device = torch.device("cuda")

        model_config = cfg.model_cfg
        model_config.device_8bit = gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()

        # OPERA convention: eval processor WITHOUT normalization, then an
        # explicit Normalize on the way in. Keeps pixels identical to the
        # numbers OPERA/HALC report.
        processor_cfg = cfg.get_config().preprocess
        processor_cfg.vis_processor.eval.do_normalize = False
        self.vis_processors, self.txt_processors = load_preprocess(
            processor_cfg)
        self.norm = transforms.Normalize(_MEAN, _STD)

        # msrc.loop counts tokens with mm.tokenizer(...)
        # MiniGPT-4's maybe_autocast() uses torch.cuda.amp.autocast, which is
        # deprecated and does NOT reliably enable autocast on torch>=2.1x.
        # Without it the fp16 ViT receives float32 pixels -> "Input type
        # (float) and bias type (c10::Half)". Rebind to the modern API so
        # every fp16/fp32 boundary in the model is handled, not just conv.
        import contextlib as _ctxlib

        def _maybe_autocast(dtype=torch.float16, _dev=self.device):
            if _dev.type == "cpu":
                return _ctxlib.nullcontext()
            return torch.amp.autocast("cuda", dtype=dtype)

        self.model.maybe_autocast = _maybe_autocast

        self.tokenizer = self.model.llama_tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id or 2
        self.pad_token_id = self.tokenizer.pad_token_id or 0

        # path -> (img_embeds, atts) cache; big enough for 1 image + 5 crops
        self._cache = {}
        self._cache_order = []
        self._cache_size = max(2, int(cache_size))

        self._resolve_yes_no()

    # ------------------------------------------------------------------
    def _resolve_yes_no(self):
        """Robust yes/no token-id resolution (POPE-parity with qwen_base).

        LLaMA/Vicuna prepends BOS, so add_special_tokens=False is mandatory
        here — otherwise every 'first token' would be <s>.
        """
        tok = self.tokenizer

        def first(words):
            out = set()
            for w in words:
                try:
                    ids = tok(w, add_special_tokens=False).input_ids
                except Exception:
                    continue
                if ids:
                    out.add(ids[0])
            return out

        yes, no = set(), set()
        try:
            vocab = tok.get_vocab()
            for _, tid in vocab.items():
                try:
                    d = tok.decode([tid]).strip().lower()
                except Exception:
                    continue
                if d == "yes":
                    yes.add(tid)
                elif d == "no":
                    no.add(tid)
        except Exception:
            pass
        yes |= first(["Yes", "yes", "▁Yes", "▁yes", "YES"])
        no |= first(["No", "no", "▁No", "▁no", "NO"])
        self.yes_ids, self.no_ids = sorted(yes - no), sorted(no - yes)
        assert self.yes_ids and self.no_ids, \
            "[minigpt4-base] could not resolve yes/no token ids"
        print(f"[minigpt4-base] yes={self.yes_ids} no={self.no_ids} "
              f"eos={self.eos_token_id}", flush=True)

    # ------------------------------------------------------------------
    def _prep_image(self, image_path):
        """MSRC image handle == the filesystem PATH (see module docstring).
        Encoding happens lazily in _img_embeds and is cached."""
        return image_path

    @torch.inference_mode()
    def _img_embeds(self, image_path):
        """PIL -> vis_processor -> normalize -> EVA-ViT -> Q-Former ->
        llama_proj. Cached per path (LRU) so TTA crops are encoded once."""
        if image_path in self._cache:
            return self._cache[image_path]
        from PIL import Image
        raw = Image.open(image_path).convert("RGB")
        px = self.vis_processors["eval"](raw).unsqueeze(0).to(self.device)
        px = self.norm(px)
        # The ViT is loaded in fp16 (vit_precision) but the processor emits
        # float32. MiniGPT-4 leans on maybe_autocast() to bridge that, which
        # uses the deprecated torch.cuda.amp.autocast and does NOT reliably
        # cast the patch-embed conv input on torch>=2.1x -> "Input type
        # (float) and bias type (c10::Half) should be the same". Match the
        # encoder dtype explicitly instead of depending on autocast.
        _vit_dtype = self.model.visual_encoder.patch_embed.proj.weight.dtype
        px = px.to(dtype=_vit_dtype)
        with self.model.maybe_autocast():
            embeds, atts = self.model.encode_img(px)
        self._cache[image_path] = (embeds, atts)
        self._cache_order.append(image_path)
        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return embeds, atts

    # ------------------------------------------------------------------
    def _context_emb(self, image_path, user_text, assistant_prefix=None):
        """Assemble inputs_embeds for one turn.

        image_path=None -> TEXT-ONLY (language prior): the <Img> block is
        dropped entirely, so no visual tokens reach the LLM.
        """
        if image_path is None:
            prompt = PROMPT_TEMPLATE_TEXTONLY.format(user_text)
            img_list = []
        else:
            prompt = PROMPT_TEMPLATE.format(user_text)
            embeds, _ = self._img_embeds(image_path)
            img_list = [embeds]
        if assistant_prefix:
            # continue the assistant turn with already-accepted caption text
            prompt = prompt + " " + assistant_prefix.strip()

        segs = prompt.split("<ImageHere>")
        assert len(segs) == len(img_list) + 1, \
            "image-placeholder / image count mismatch"
        seg_ids = [
            self.tokenizer(s, return_tensors="pt",
                           add_special_tokens=(i == 0)).to(
                               self.device).input_ids
            for i, s in enumerate(segs)
        ]
        seg_embs = [self.model.embed_tokens(t) for t in seg_ids]
        mixed = [e for pair in zip(seg_embs[:-1], img_list) for e in pair]
        mixed = mixed + [seg_embs[-1]]
        # Q-Former and llama_proj stay fp32, so image embeds are fp32 while
        # embed_tokens yields fp16 text embeds. torch.cat promotes the mix to
        # fp32, which then hits fp16 LLaMA weights -> "self and mat2 must have
        # the same dtype". Force the whole sequence to the LLM's dtype.
        _llm_dtype = self.model.llama_model.get_input_embeddings().weight.dtype
        return torch.cat([m.to(_llm_dtype) for m in mixed], dim=1)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def yes_no_gap(self, user_msg, image_path=None):
        """logsumexp(yes) - logsumexp(no) at the first answer position.
        image_path=None -> text-only pass (language prior, no image).
        sigmoid(gap) == p(yes)/(p(yes)+p(no)), so verify() == sigmoid of
        this and debiasing is a subtraction in gap space."""
        embs = self._context_emb(image_path, user_msg)
        attn = torch.ones(embs.shape[:2], dtype=torch.long,
                          device=self.device)
        with self.model.maybe_autocast():
            logits = self.model.llama_model(
                inputs_embeds=embs, attention_mask=attn).logits[0, -1].float()
        y = torch.logsumexp(torch.stack([logits[i] for i in self.yes_ids]), 0)
        n = torch.logsumexp(torch.stack([logits[i] for i in self.no_ids]), 0)
        return (y - n).item()

    @torch.inference_mode()
    def verify(self, question, image_path):
        g = self.yes_no_gap(question + VERIF_SUFFIX, image_path)
        return 1.0 / (1.0 + math.exp(-g))

    # ------------------------------------------------------------------
    class _SentenceStop(StoppingCriteria):
        """Stop at the first sentence boundary or at MiniGPT-4's end_sym.

        With inputs_embeds, transformers-4.29.2 seeds input_ids with a lone
        BOS, so decoding with skip_special_tokens=True yields exactly the
        newly generated text.
        """

        def __init__(self, tokenizer, min_new=4):
            self.tokenizer = tokenizer
            self.min_new = min_new

        def __call__(self, output_ids, scores, **kwargs):
            if output_ids.shape[1] < self.min_new:
                return False
            text = self.tokenizer.decode(output_ids[0].tolist(),
                                         skip_special_tokens=True)
            return bool(_SENT_END.search(text)) or "###" in text

    @torch.inference_mode()
    def generate_sentence(self, image_path, assistant_prefix,
                          max_chunk_tokens=90):
        """Greedy-decode the next sentence, continuing `assistant_prefix`.
        Returns (text, hit_eos) exactly like the Qwen lane."""
        embs = self._context_emb(image_path, CAPTION_PROMPT,
                                 assistant_prefix=assistant_prefix)
        attn = torch.ones(embs.shape[:2], dtype=torch.long,
                          device=self.device)
        stop = StoppingCriteriaList([self._SentenceStop(self.tokenizer)])
        with self.model.maybe_autocast():
            out = self.model.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_chunk_tokens,
                min_length=1,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                stopping_criteria=stop,
                use_cache=True,
            )
        ids = out[0].tolist()
        if ids and ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]                       # strip the seeded BOS
        hit_eos = self.eos_token_id in ids
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        if "###" in text:                       # MiniGPT-4's end_sym
            text = text.split("###")[0]
            hit_eos = True
        text = text.replace("</s>", "").replace("<s>", "")
        return text.strip(), hit_eos

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_short(self, image_path, question, max_new_tokens=6):
        """Short free-form answer (used by the re-grounding substitute)."""
        embs = self._context_emb(image_path, question)
        attn = torch.ones(embs.shape[:2], dtype=torch.long,
                          device=self.device)
        with self.model.maybe_autocast():
            out = self.model.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                min_length=1,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                use_cache=True,
            )
        ids = out[0].tolist()
        if ids and ids[0] == self.tokenizer.bos_token_id:
            ids = ids[1:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text.split("###")[0].strip()
