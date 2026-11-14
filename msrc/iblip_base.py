# -*- coding: utf-8 -*-
"""
InstructBLIP (Vicuna-7B) as the MSRC BASE MODEL — generator, self-verifier and
re-grounder. Third-architecture lane; drop-in sibling of msrc/qwen_base.py.

CANONICAL VERSION. Every hot-patch from the 2026-07-24 debugging session is
baked in here so you never sed this file again:

  P1  _encode() encodes the image, the LM text and the Q-Former text
      SEPARATELY. InstructBlipProcessor.__call__ does `img_encoding +
      txt_encoding` internally, which is `list + Tensor` on the transformers
      build in the `modality` env and raises
        TypeError: can only concatenate list (not "Tensor") to list
      That single bug is what produced the wall of `[gate] FAILED ...` lines
      and the vacuous NO-GO with object_claims=0.

  P2  The LM prompt is prefixed with num_query_tokens copies of the image
      token, which the combined processor would have inserted for us. Without
      it, modeling_instructblip.generate() hits
        RuntimeError: shape mismatch: value tensor of shape [131072] cannot be
        broadcast to indexing result of shape [0]
      because it finds zero image placeholders to write Q-Former output into.

  P3  generate_sentence() generates the FULL caption ONCE with a 512-token
      budget and a repetition penalty, then serves the MSRC loop one sentence
      at a time from a per-image cache. InstructBLIP has no chat template,
      ignores "continue this description exactly where it stops" prompts, and
      hits EOS after ~17 words — so the sentence-at-a-time prefix-continuation
      that works for LLaVA and Qwen yields one-sentence captions no matter how
      large max_new_tokens is. max_new_tokens is a BUDGET, not a target.

  P4  No prompt-slicing anywhere. HF InstructBLIP's generate() returns ONLY
      the new tokens, so out[0] is decoded directly and <s>/</s>/<pad> are
      stripped textually rather than by index arithmetic.

Duck-typed to msrc.loop.msrc_caption and msrc.reground_loop.caption_fixed:
    _prep_image(path)                 -> PIL.Image handle
    verify(question, handle)          -> p(Yes)/(p(Yes)+p(No)) in [0,1]
    generate_sentence(handle, prefix) -> (text, hit_eos)
    .tokenizer                        -> for token counting
plus module-level:
    reground_iblip(mm, handle, head)  -> (substitute_word, v_sub)

No llava imports. No trust_remote_code. No Qwen dependency.
"""
import re
import torch
from PIL import Image
from transformers import (InstructBlipProcessor,
                          InstructBlipForConditionalGeneration)

# Kept byte-identical to the LLaVA/Qwen lanes so the verifier sees the same
# question surface across all three backbones. Do not "improve" this string
# without re-running calibration on all three lanes.
VERIF_SUFFIX = "\nAnswer the question using a single word or phrase."

# P3: InstructBLIP responds to an explicit length request. This is the prompt
# that produced fluent multi-sentence captions in the 2026-07-24 smoke test.
CAPTION_PROMPT = ("Describe this image in detail. Write a long thorough "
                  "paragraph covering every object, person, color, and "
                  "activity visible.")

YES_WORDS = ["Yes", "yes", "YES"]
NO_WORDS = ["No", "no", "NO"]


def _first_ids(tokenizer, words):
    """Leading content-token id for each surface form, deduped, order-stable."""
    ids = []
    for w in words:
        for enc in (tokenizer(w, add_special_tokens=False).input_ids,
                    tokenizer(" " + w, add_special_tokens=False).input_ids):
            if not enc:
                continue
            tid = enc[0]
            # LLaMA/Vicuna emits a bare SP underscore token for the space
            # variant; skip it and take the next id if so.
            if len(enc) > 1 and tokenizer.convert_ids_to_tokens([tid])[0] in ("▁", "<0x20>"):
                tid = enc[1]
            if tid not in ids:
                ids.append(tid)
    return ids


class IBlipBaseModel(object):
    """InstructBLIP-Vicuna-7B wrapped to the MSRC base-model contract."""

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.processor = InstructBlipProcessor.from_pretrained(model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
        self.model.eval()

        # The LM tokenizer is the one the MSRC loop counts tokens with.
        self.tokenizer = self.processor.tokenizer
        self.qformer_tokenizer = getattr(self.processor,
                                         "qformer_tokenizer", None)
        if self.qformer_tokenizer is None:
            raise RuntimeError(
                "processor has no qformer_tokenizer; your transformers build "
                "is too old for the separate-encode path. Run:\n"
                "  python -c \"from transformers import InstructBlipProcessor;"
                "p=InstructBlipProcessor.from_pretrained('<path>');"
                "print([a for a in dir(p) if 'token' in a])\"\n"
                "and paste the output.")

        self.yes_ids = _first_ids(self.tokenizer, YES_WORDS)
        self.no_ids = _first_ids(self.tokenizer, NO_WORDS)
        self.eos_id = self.tokenizer.eos_token_id

        # P2: how many image placeholders the LM prompt must carry.
        self.n_img = int(getattr(self.model.config, "num_query_tokens", 32))
        itok = getattr(self.processor, "image_token", None)
        self.image_token = getattr(itok, "content", itok) or "<image>"

        # P3: per-image sentence cache, keyed on handle identity.
        self._cache = None

        print(f"[iblip-base] yes={self.yes_ids} no={self.no_ids} "
              f"eos={self.eos_id} n_img={self.n_img} "
              f"image_token={self.image_token!r}", flush=True)

    VERIF_SUFFIXES = [VERIF_SUFFIX]

    # ------------------------------------------------------------------ image
    def _prep_image(self, path):
        """The MSRC loop holds this handle for the whole image, so identity is
        stable and the P3 sentence cache can key on it."""
        return Image.open(path).convert("RGB")

    # ----------------------------------------------------------------- encode
    def _encode(self, image, text):
        """P1 + P2. Encode the three streams separately and merge by hand,
        bypassing InstructBlipProcessor.__call__'s broken list+Tensor concat,
        and prepend the image placeholders the combined path would have added.
        """
        img = self.processor.image_processor(images=image, return_tensors="pt")
        # P2: image placeholders belong ONLY in the LM stream, never in the
        # Q-Former stream (the Q-Former reads the raw instruction text).
        lm_text = (self.image_token * self.n_img) + text
        txt = self.tokenizer(text=lm_text, return_tensors="pt")
        qf = self.qformer_tokenizer(text=text, return_tensors="pt")
        return {
            "pixel_values": img["pixel_values"].to(self.device, self.dtype),
            "input_ids": txt["input_ids"].to(self.device),
            "attention_mask": txt["attention_mask"].to(self.device),
            "qformer_input_ids": qf["input_ids"].to(self.device),
            "qformer_attention_mask": qf["attention_mask"].to(self.device),
        }

    # ----------------------------------------------------------------- verify
    @torch.inference_mode()
    def verify(self, question, image):
        """One forward pass -> V = p(Yes)/(p(Yes)+p(No)) in [0,1].

        Identical functional form to the LLaVA and Qwen verifiers, so tau
        thresholds are on the same scale across lanes. (The logit-gap +
        sigmoid variant used in the POPE lanes is a monotone reparametrisation
        of this and yields the same value; keep this form here so CHAIR
        calibration stays comparable.)
        """
        enc = self._encode(image, question + VERIF_SUFFIX)
        out = self.model(**enc)
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        py = sum(probs[i].item() for i in self.yes_ids)
        pn = sum(probs[i].item() for i in self.no_ids)
        return py / max(py + pn, 1e-9)

    @torch.inference_mode()
    def yes_no_gap(self, question, image):
        """Raw logit gap max(yes) - max(no). Not used by the CHAIR lane; kept
        so offline tau/alpha sweeps can reuse this adapter without a rerun."""
        enc = self._encode(image, question + VERIF_SUFFIX)
        logits = self.model(**enc).logits[0, -1].float()
        return (max(logits[i].item() for i in self.yes_ids)
                - max(logits[i].item() for i in self.no_ids))

    # --------------------------------------------------------------- generate
    @staticmethod
    def _clean(text):
        for junk in ("<s>", "</s>", "<pad>"):
            text = text.replace(junk, "")
        return text.strip()

    @torch.inference_mode()
    def generate_sentence(self, image, assistant_prefix,
                          max_chunk_tokens=512):
        """P3. First call for an image (assistant_prefix falsy) generates the
        WHOLE caption and caches its sentences; subsequent calls pop the next
        sentence. Returns (text, hit_eos) where hit_eos signals the loop that
        the caption is finished.

        Why not prefix-continuation like LLaVA/Qwen: InstructBLIP has no chat
        template and ignores continuation instructions, so asking it to resume
        after a prefix returns either a restart or an immediate EOS. Generating
        once and splitting keeps the loop's sentence-wise contract intact
        while letting the model write at its natural length.

        Consequence worth stating in the paper: unlike LLaVA/Qwen, downstream
        sentences are NOT re-conditioned on the repaired prefix. Deletions and
        substitutions still happen mid-caption, but the model does not get to
        continue "as though it had said lamp all along". This is a real
        architectural limitation of the backbone, not a design choice.
        """
        if assistant_prefix:
            cache = getattr(self, "_cache", None)
            if cache is not None and cache[0] is image and cache[1]:
                nxt = cache[1].pop(0)
                return nxt, (len(cache[1]) == 0)
            # Cache miss or exhausted -> caption is over. Terminating here is
            # correct and cannot loop: the loop stops on hit_eos.
            return "", True

        enc = self._encode(image, CAPTION_PROMPT)
        out = self.model.generate(
            **enc, do_sample=False, num_beams=1,
            max_new_tokens=max_chunk_tokens,
            repetition_penalty=1.15, length_penalty=1.0, use_cache=True)
        text = self._clean(self.tokenizer.decode(out[0],
                                                 skip_special_tokens=True))

        # Local import: msrc.lattice is pure-python, no torch, no llava.
        from msrc.lattice import split_sentences
        sents = [s for s in split_sentences(text) if len(s.split()) >= 3]
        self._cache = (image, sents[1:])
        return (sents[0] if sents else ""), (len(sents) <= 1)


# ------------------------------------------------------- InstructBLIP reground
REGROUND_Q = ("There may or may not be a {head} in the image. What similar "
              "object is clearly visible instead? Answer with a single word.")


@torch.inference_mode()
def reground_iblip(mm, image, head, verifier=None):
    """Signature matches reground_qwen(mm, image_path, head, verifier=None).
    -> (word or None, v_sub, g_sub, g_sub_prior). Substitute is generated from
    the ORIGINAL image; only verification runs through the (possibly TTA'd)
    verifier, so v_sub is on the same scale tau_sub was calibrated on."""
    from msrc.tta import TTAVerifier
    if verifier is None:
        verifier = TTAVerifier(mm)
    if isinstance(image, str):
        image = mm._prep_image(image)
    q = REGROUND_Q.format(head=head)
    enc = mm._encode(image, q)
    out = mm.model.generate(**enc, do_sample=False, num_beams=1,
                            max_new_tokens=6, use_cache=True)
    w = mm._clean(mm.tokenizer.decode(out[0], skip_special_tokens=True))
    w = w.split()[0].strip(".,").lower() if w.split() else ""
    try:
        from msrc.reground_loop import singularize
        w = singularize(w)
    except Exception:
        pass
    if not w or not w.isalpha() or w == head.lower():
        return None, 0.0, 0.0, 0.0
    art = "an" if w[0] in "aeiou" else "a"
    v, g, gp = verifier.score(f"Is there {art} {w} in the image?", image)
    return w, v, g, gp
