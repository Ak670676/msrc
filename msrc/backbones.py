# -*- coding: utf-8 -*-
"""
One interface, four backbones, ONE protocol.

Every backbone is driven through exactly three calls, with exactly the same
prompt strings and the same decoding settings:

    caption(path)            -> str    full caption, one greedy generate()
    gap(user_text, path)     -> float  logsumexp(yes) - logsumexp(no) at the
                                       first answer position
    short(user_text, path)   -> str    greedy answer, <= SHORT_TOKENS tokens

V = sigmoid(gap) == p(yes)/(p(yes)+p(no)), identical functional form on every
backbone, so thresholds live on the same scale.

What is shared (the protocol):
  * CAPTION_PROMPT, VERIF_SUFFIX, SUB_PROMPT (below)
  * greedy, num_beams=1, max_new_tokens=512, repetition_penalty=1.0
    (no backbone-specific penalty: Qwen2.5 ships 1.05 and the old
    InstructBLIP lane used 1.15 -- both are switched off here)
  * yes/no token sets resolved by one rule: every vocab entry that decodes,
    stripped and lower-cased, to exactly "yes" / "no"
  * each backbone's own native chat template (the only thing that differs,
    and it has to: it is part of the model)

The old per-backbone adapters are imported only for model loading and their
low-level encoders; none of their prompts or decoding settings are used.
"""
import math
import os
import re

import torch

CAPTION_PROMPT = "Describe this image in detail."
VERIF_SUFFIX = "\nAnswer the question using a single word or phrase."
SUB_PROMPT = ("There may or may not be a {head} in the image. What similar "
              "object is clearly visible instead? Answer with a single word.")
MAX_NEW_TOKENS = 512
SHORT_TOKENS = 12

_M = os.environ.get("MSRC_MODELS", "models")      # directory holding the checkpoints
MODEL_PATHS = {
    "llava": os.path.join(_M, "llava-v1.5-7b"),
    "qwen25": os.path.join(_M, "Qwen2.5-VL-3B-Instruct"),
    "qwen2vl": os.path.join(_M, "Qwen2-VL-7B-Instruct"),
    "iblip": os.path.join(_M, "instructblip-vicuna-7b"),
    "minigpt4": "eval_configs/minigpt4_msrc_eval.yaml",
}


def yes_no_ids(tokenizer):
    yes, no = set(), set()
    for tid in range(len(tokenizer)):
        try:
            d = tokenizer.decode([tid]).strip().lower()
        except Exception:
            continue
        if d == "yes":
            yes.add(tid)
        elif d == "no":
            no.add(tid)
    assert yes and no, "could not resolve yes/no ids"
    return sorted(yes), sorted(no)


def _lse_gap(logits, yes, no):
    logits = logits.float()
    y = torch.logsumexp(logits[yes.to(logits.device)], 0)
    n = torch.logsumexp(logits[no.to(logits.device)], 0)
    return (y - n).item()


def sigmoid(g):
    return 1.0 / (1.0 + math.exp(-g))


class _Base:
    name = "?"

    def _setup_ids(self, tokenizer):
        y, n = yes_no_ids(tokenizer)
        self.yes_ids, self.no_ids = y, n
        self._yes = torch.tensor(y)
        self._no = torch.tensor(n)
        print(f"[{self.name}] yes={y} no={n}", flush=True)

    def verify(self, question, path):
        return sigmoid(self.gap(question + VERIF_SUFFIX, path))


# --------------------------------------------------------------- LLaVA-1.5
class LLaVA(_Base):
    name = "llava"

    def __init__(self, model_path=MODEL_PATHS["llava"]):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        self.tok, self.model, self.improc, _ = load_pretrained_model(
            model_path, None, get_model_name_from_path(model_path),
            device="cuda")
        self.model.eval()
        self.model.generation_config.repetition_penalty = 1.0
        self._setup_ids(self.tok)
        self._img = (None, None)

    def _image(self, path):
        if self._img[0] != path:
            from PIL import Image
            from llava.mm_utils import process_images
            t = process_images([Image.open(path).convert("RGB")],
                               self.improc, self.model.config)
            self._img = (path, t.to("cuda", dtype=torch.float16))
        return self._img[1]

    def _ids(self, text):
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        return tokenizer_image_token(conv.get_prompt(), self.tok,
                                     IMAGE_TOKEN_INDEX,
                                     return_tensors="pt").unsqueeze(0).cuda()

    @torch.inference_mode()
    def gap(self, text, path):
        out = self.model(self._ids(text), images=self._image(path),
                         use_cache=False)
        return _lse_gap(out.logits[0, -1], self._yes, self._no)

    @torch.inference_mode()
    def _gen(self, text, path, n):
        ids = self._ids(text)
        out = self.model.generate(ids, images=self._image(path),
                                  do_sample=False, num_beams=1,
                                  max_new_tokens=n, repetition_penalty=1.0,
                                  use_cache=True)
        new = out[0]
        if out.shape[1] > ids.shape[1] and torch.equal(out[0, :4], ids[0, :4]):
            new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()

    def caption(self, path):
        return self._gen(CAPTION_PROMPT, path, MAX_NEW_TOKENS)

    def short(self, text, path):
        return self._gen(text, path, SHORT_TOKENS)


# ------------------------------------------------------------ Qwen2.5-VL-3B
class Qwen25(_Base):
    name = "qwen25"

    def __init__(self, model_path=MODEL_PATHS["qwen25"], model_cls=None):
        from msrc.qwen_base import QwenBaseModel
        self.mm = QwenBaseModel(model_path, model_cls)
        self.mm.model.generation_config.repetition_penalty = 1.0
        self._setup_ids(self.mm.tokenizer)
        # Exact speed-up: the vision tower's output depends only on the
        # image, and its windowed attention loops with a host sync per window
        # (~2 s/call on a shared GPU). Memoise it per image path.
        visual = self.mm.model.visual
        orig = visual.forward
        self._cur, self._vcache = None, (None, None)

        def cached(pixel_values, grid_thw=None, **kw):
            if self._cur is not None and self._vcache[0] == self._cur:
                return self._vcache[1]
            out = orig(pixel_values, grid_thw=grid_thw, **kw)
            self._vcache = (self._cur, out)
            return out
        visual.forward = cached

    def _ctx(self, path, text):
        self._cur = path
        return self.mm._context(path, text)

    @torch.inference_mode()
    def gap(self, text, path):
        logits = self.mm.model(**self._ctx(path, text)).logits[0, -1]
        return _lse_gap(logits, self._yes, self._no)

    @torch.inference_mode()
    def _gen(self, text, path, n):
        ctx = self._ctx(path, text)
        out = self.mm.model.generate(
            **ctx, do_sample=False, num_beams=1, max_new_tokens=n,
            repetition_penalty=1.0, eos_token_id=self.mm.im_end_id,
            pad_token_id=self.mm.im_end_id, use_cache=True)
        new = [t for t in out[0, ctx["input_ids"].shape[1]:].tolist()
               if t not in self.mm.stop_ids]
        return self.mm._clean(self.mm.tokenizer.decode(new))

    def caption(self, path):
        return self._gen(CAPTION_PROMPT, path, MAX_NEW_TOKENS)

    def short(self, text, path):
        return self._gen(text, path, SHORT_TOKENS)


# ------------------------------------------------------------ Qwen2-VL-7B
class Qwen2VL(Qwen25):
    """Qwen2-VL-7B-Instruct (the model CEBC reports). Same chat template,
    processor path and decoding as Qwen2.5; only the model class differs."""
    name = "qwen2vl"

    def __init__(self, model_path=MODEL_PATHS["qwen2vl"]):
        from transformers import Qwen2VLForConditionalGeneration
        super().__init__(model_path, Qwen2VLForConditionalGeneration)


# --------------------------------------------------- InstructBLIP-Vicuna-7B
class InstructBLIP(_Base):
    name = "iblip"

    def __init__(self, model_path=MODEL_PATHS["iblip"]):
        from msrc.iblip_base import IBlipBaseModel
        self.mm = IBlipBaseModel(model_path)
        self._setup_ids(self.mm.tokenizer)
        self._img = (None, None)

    def _image(self, path):
        if self._img[0] != path:
            from PIL import Image
            self._img = (path, Image.open(path).convert("RGB"))
        return self._img[1]

    @torch.inference_mode()
    def gap(self, text, path):
        out = self.mm.model(**self.mm._encode(self._image(path), text))
        return _lse_gap(out.logits[0, -1], self._yes, self._no)

    @torch.inference_mode()
    def _gen(self, text, path, n):
        enc = self.mm._encode(self._image(path), text)
        out = self.mm.model.generate(**enc, do_sample=False, num_beams=1,
                                     max_new_tokens=n,
                                     repetition_penalty=1.0,
                                     length_penalty=1.0, use_cache=True)
        return self.mm._clean(self.mm.tokenizer.decode(
            out[0], skip_special_tokens=True))

    def caption(self, path):
        return self._gen(CAPTION_PROMPT, path, MAX_NEW_TOKENS)

    def short(self, text, path):
        return self._gen(text, path, SHORT_TOKENS)


# ------------------------------------------------------------- MiniGPT-4 7B
class MiniGPT4(_Base):
    name = "minigpt4"

    def __init__(self, cfg_path=MODEL_PATHS["minigpt4"]):
        from msrc.minigpt4_base import MiniGPT4BaseModel
        self.mm = MiniGPT4BaseModel(cfg_path=cfg_path)
        self._setup_ids(self.mm.tokenizer)

    @torch.inference_mode()
    def gap(self, text, path):
        embs = self.mm._context_emb(path, text)
        attn = torch.ones(embs.shape[:2], dtype=torch.long,
                          device=self.mm.device)
        with self.mm.model.maybe_autocast():
            logits = self.mm.model.llama_model(
                inputs_embeds=embs, attention_mask=attn).logits[0, -1]
        return _lse_gap(logits, self._yes, self._no)

    class _Stop:
        """Stop on MiniGPT-4's end_sym '###' (its turn separator)."""

        def __init__(self, tok):
            self.tok = tok

        def __call__(self, output_ids, scores, **kw):
            return "###" in self.tok.decode(output_ids[0, -4:].tolist(),
                                            skip_special_tokens=True)

    @torch.inference_mode()
    def _gen(self, text, path, n):
        from transformers import StoppingCriteriaList
        embs = self.mm._context_emb(path, text)
        attn = torch.ones(embs.shape[:2], dtype=torch.long,
                          device=self.mm.device)
        with self.mm.model.maybe_autocast():
            out = self.mm.model.llama_model.generate(
                inputs_embeds=embs, attention_mask=attn, do_sample=False,
                num_beams=1, max_new_tokens=n, repetition_penalty=1.0,
                eos_token_id=self.mm.eos_token_id,
                pad_token_id=self.mm.pad_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [self._Stop(self.mm.tokenizer)]),
                use_cache=True)
        ids = out[0].tolist()
        if ids and ids[0] == self.mm.tokenizer.bos_token_id:
            ids = ids[1:]
        t = self.mm.tokenizer.decode(ids, skip_special_tokens=True)
        return t.split("###")[0].replace("</s>", "").strip()

    def caption(self, path):
        return self._gen(CAPTION_PROMPT, path, MAX_NEW_TOKENS)

    def short(self, text, path):
        return self._gen(text, path, SHORT_TOKENS)


# ------------------------------------------------- LLaVA-1.5-7B (llava-hf)
class LLaVAHF(_Base):
    """llava-hf/llava-1.5-7b-hf through transformers, with the exact prompt
    template the decoding baselines use (decode_baselines.TEMPLATE), so MSRC
    and the baselines share one checkpoint and one code path."""
    name = "llava_hf"
    TEMPLATE = "USER: <image>\n{q} ASSISTANT:"

    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf"):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16).to("cuda").eval()
        self._setup_ids(self.proc.tokenizer)
        self._img = (None, None)

    def _enc(self, text, path):
        if self._img[0] != path:
            from PIL import Image
            self._img = (path, Image.open(path).convert("RGB"))
        enc = self.proc(images=self._img[1],
                        text=self.TEMPLATE.format(q=text), return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(torch.float16)
        return enc

    @torch.inference_mode()
    def gap(self, text, path):
        out = self.model(**self._enc(text, path), use_cache=False)
        return _lse_gap(out.logits[0, -1], self._yes, self._no)

    @torch.inference_mode()
    def _gen(self, text, path, n):
        enc = self._enc(text, path)
        out = self.model.generate(**enc, do_sample=False, num_beams=1,
                                  max_new_tokens=n, repetition_penalty=1.0,
                                  use_cache=True)
        return self.proc.tokenizer.decode(
            out[0, enc["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

    def caption(self, path):
        return self._gen(CAPTION_PROMPT, path, MAX_NEW_TOKENS)

    def short(self, text, path):
        return self._gen(text, path, SHORT_TOKENS)


BACKBONES = {"llava": LLaVA, "llava_hf": LLaVAHF, "qwen25": Qwen25, "qwen2vl": Qwen2VL,
             "iblip": InstructBLIP,
             "minigpt4": MiniGPT4}
ENVS = {"llava": "modality", "llava_hf": "qwen25", "qwen25": "qwen25", "qwen2vl": "qwen25",
        "iblip": "modality",
        "minigpt4": "modality"}


def load(name):
    return BACKBONES[name]()
