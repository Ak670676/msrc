# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-3B-Instruct as the MSRC BASE MODEL (generator + self-verifier).

Port of the Qwen-VL-Chat wrapper (kept as qwen_base.py.qwenvlchat_original)
to the transformers-native Qwen2.5-VL API. The METHOD is unchanged: only the
backbone is swapped, so any difference in the final numbers is attributable
to the model and nothing else.

What changed vs the Qwen-VL-Chat wrapper
----------------------------------------
Qwen-VL-Chat embedded the image PATH inside the chatml token stream, so a
"context" was a plain list of token ids and `model(ids)` was enough.
Qwen2.5-VL uses a processor that emits `pixel_values` + `image_grid_thw`
alongside `input_ids`, so a context is now a dict of model inputs.

  _context(path, text)          -> dict of model inputs  (was: list[int])
  generate_from_context(ctx, n) -> decoded string        (new helper)

Both re-grounding functions call those two instead of building token tensors
by hand. Everything else in the duck-typed contract is byte-compatible:

  _prep_image(path)                    -> the path (crop TTA needs paths)
  yes_no_gap(user_msg, image_path)     -> logsumexp(yes) - logsumexp(no)
  verify(question, image_path)         -> sigmoid(gap)
  generate_sentence(path, prefix)      -> (text, hit_eos)
  .tokenizer / .model / .im_end_id / .VERIF_SUFFIXES
"""
import math
import os
import re

import torch
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          StoppingCriteria, StoppingCriteriaList)

VERIF_SUFFIX = "\nAnswer the question using a single word or phrase."
CAPTION_PROMPT = "Describe the image in detail."
SYSTEM = "You are a helpful assistant."

# Bound the visual token budget. COCO val2014 is ~640x480, and the crop-TTA
# views are smaller again, so this cap never truncates a real image -- it
# only stops an unusually large file from blowing up latency.
MIN_PIXELS = int(os.environ.get("QWEN25_MIN_PIXELS", 256 * 28 * 28))
MAX_PIXELS = int(os.environ.get("QWEN25_MAX_PIXELS", 1280 * 28 * 28))


class QwenBaseModel:
    # Suffix templates for prompt ensembling (index 0 = POPE-standard).
    VERIF_SUFFIXES = [
        VERIF_SUFFIX,
        "\nPlease answer yes or no.",
        "",
    ]

    def __init__(self, model_path, model_cls=None):
        self.processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self.tokenizer = self.processor.tokenizer
        self.model = (model_cls or Qwen2_5_VLForConditionalGeneration).from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        ).eval()

        # Greedy everywhere. Qwen2.5-VL ships do_sample=true (with
        # temperature 1e-6); the Qwen-VL-Chat wrapper forced sampling off at
        # config level and we keep that guard rather than trusting every call
        # site to pass do_sample=False. top_k/top_p are left unset -- they are
        # sampling-only knobs and setting them here just emits a warning on
        # every generate() call. repetition_penalty=1.05 is kept as shipped:
        # it is part of the model's own greedy decoding policy and is
        # deterministic, so reproducibility is unaffected.
        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        except Exception:
            pass

        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.stop_ids = {i for i in (self.im_end_id, self.eot_id)
                         if i is not None and i >= 0}

        # ---- yes / no token ids -------------------------------------------
        # Same resolution strategy as the Qwen-VL-Chat wrapper: scan the whole
        # vocabulary for tokens that decode to bare "yes"/"no", then union the
        # first-token ids of the obvious surface forms.
        def first(ws):
            s = set()
            for w in ws:
                ids = self.tokenizer(w, add_special_tokens=False).input_ids
                if ids:
                    s.add(ids[0])
            return s

        yes, no = set(), set()
        try:
            for tok, tid in self.tokenizer.get_vocab().items():
                try:
                    d = self.tokenizer.decode([tid]).strip().lower()
                except Exception:
                    continue
                if d == "yes":
                    yes.add(tid)
                elif d == "no":
                    no.add(tid)
        except Exception:
            pass
        yes |= first(["Yes", "yes", " Yes", " yes", "YES"])
        no |= first(["No", "no", " No", " no", "NO"])
        self.yes_ids, self.no_ids = sorted(yes - no), sorted(no - yes)
        assert self.yes_ids and self.no_ids, \
            "[qwen25] could not resolve yes/no token ids"
        self._yes_idx = torch.tensor(self.yes_ids)
        self._no_idx = torch.tensor(self.no_ids)
        print(f"[qwen25] {os.path.basename(model_path)} | "
              f"yes={self.yes_ids} no={self.no_ids} im_end={self.im_end_id} "
              f"| pixels {MIN_PIXELS}..{MAX_PIXELS}", flush=True)

    # ------------------------------------------------------------------
    def _prep_image(self, image_path):
        # Crop TTA writes JPEG views and passes their paths, so the handle
        # stays a path exactly as in the Qwen-VL-Chat lane.
        return image_path

    def _messages(self, image_path, user_text):
        content = []
        if image_path is not None:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]

    def _context(self, image_path, user_text, assistant_prefix=None):
        """Model inputs for one turn.

        image_path=None -> text-only pass (the language-prior term the TTA
        verifier subtracts when alpha > 0).
        """
        msgs = self._messages(image_path, user_text)
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        if assistant_prefix:
            # The template ends right after "<|im_start|>assistant\n"; append
            # the already-accepted caption text and let the model continue it.
            text = text + " " + assistant_prefix.strip()

        images = None
        if image_path is not None:
            from qwen_vl_utils import process_vision_info
            images, _ = process_vision_info(msgs)

        inputs = self.processor(text=[text], images=images, videos=None,
                                padding=True, return_tensors="pt")
        return inputs.to(self.model.device)

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def yes_no_gap(self, user_msg, image_path=None):
        """logsumexp(yes) - logsumexp(no) at the first answer position.

        sigmoid(gap) == p(yes)/(p(yes)+p(no)), so verify() is a sigmoid of
        this and prior debiasing is a subtraction in gap space. Identical
        definition to the Qwen-VL-Chat lane.
        """
        inputs = self._context(image_path, user_msg)
        logits = self.model(**inputs).logits[0, -1].float()
        y = torch.logsumexp(logits[self._yes_idx.to(logits.device)], 0)
        n = torch.logsumexp(logits[self._no_idx.to(logits.device)], 0)
        return (y - n).item()

    @torch.inference_mode()
    def verify(self, question, image_path):
        g = self.yes_no_gap(question + VERIF_SUFFIX, image_path)
        return 1.0 / (1.0 + math.exp(-g))

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_from_context(self, ctx, max_new_tokens=6):
        """Greedy continuation of a context built by _context(). Used by the
        re-grounding probes, which previously hand-rolled a token tensor."""
        out = self.model.generate(
            **ctx, do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.im_end_id, pad_token_id=self.im_end_id,
            use_cache=True)
        new = out[0, ctx["input_ids"].shape[1]:].tolist()
        new = [t for t in new if t not in self.stop_ids]
        return self._clean(self.tokenizer.decode(new))

    @staticmethod
    def _clean(text):
        return (text.replace("<|im_end|>", "")
                    .replace("<|endoftext|>", "").strip())

    # ------------------------------------------------------------------
    class _SentenceStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_len):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, output_ids, scores, **kwargs):
            new = output_ids[0, self.prompt_len:]
            if new.numel() < 4:
                return False
            text = self.tokenizer.decode(new.tolist())
            return bool(re.search(r"[.!?]", text))

    @torch.inference_mode()
    def generate_sentence(self, image_path, assistant_prefix,
                          max_chunk_tokens=90):
        ctx = self._context(image_path, CAPTION_PROMPT,
                            assistant_prefix=assistant_prefix)
        plen = ctx["input_ids"].shape[1]
        stop = StoppingCriteriaList([self._SentenceStop(self.tokenizer, plen)])
        out = self.model.generate(
            **ctx, do_sample=False, num_beams=1,
            max_new_tokens=max_chunk_tokens,
            eos_token_id=self.im_end_id, pad_token_id=self.im_end_id,
            stopping_criteria=stop, use_cache=True)
        new_ids = out[0, plen:].tolist()
        hit_eos = any(t in self.stop_ids for t in new_ids)
        new_ids = [t for t in new_ids if t not in self.stop_ids]
        return self._clean(self.tokenizer.decode(new_ids)), hit_eos
