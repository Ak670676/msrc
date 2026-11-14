# -*- coding: utf-8 -*-
"""
MSRC verifier + generation loop for the official haotian-liu LLaVA codebase.

Verification: V(c) = p(Yes) / (p(Yes)+p(No)) at the first answer position,
POPE-style prompt ("Answer the question using a single word or phrase.").

Generation: greedy, one sentence at a time. After each sentence: extract
claims, verify (lazily up the ladder in 'msrc' mode), splice repairs, and
CONTINUE GENERATION CONDITIONED ON THE REPAIRED TEXT (in-loop repair).
"""
import re
import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (tokenizer_image_token, process_images,
                            get_model_name_from_path)

from msrc.lattice import (extract_claims, question_for, n_rungs,
                          repair_edit_for, apply_repairs, split_sentences)

VERIF_SUFFIX = "\nAnswer the question using a single word or phrase."
CAPTION_PROMPT = "Describe the image in detail."


class MSRCModel:
    def __init__(self, model_path, conv_mode="llava_v1", device="cuda"):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, None, model_name, device=device)
        self.model.eval()
        self.conv_mode = conv_mode
        self.device = device
        self.yes_ids, self.no_ids = self._yes_no_ids()
        print(f"[verifier] yes ids={sorted(self.yes_ids)} "
              f"no ids={sorted(self.no_ids)}")

    def _yes_no_ids(self):
        def first_ids(words):
            ids = set()
            for w in words:
                toks = self.tokenizer(w, add_special_tokens=False).input_ids
                if toks:
                    ids.add(toks[0])
            return ids
        yes = first_ids(["Yes", "yes", "YES"])
        no = first_ids(["No", "no", "NO"])
        return yes - no, no - yes

    # ------------------------------------------------------------------
    def _prep_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = process_images([image], self.image_processor,
                                self.model.config)
        return tensor.to(self.device, dtype=torch.float16)

    def _build_prompt(self, user_msg, assistant_prefix=None):
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + user_msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if assistant_prefix:
            prompt = prompt + " " + assistant_prefix
        return prompt

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def verify(self, question, image_tensor):
        """One forward pass -> V = p(Yes)/(p(Yes)+p(No)) in [0,1]."""
        prompt = self._build_prompt(question + VERIF_SUFFIX)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).to(self.device)
        out = self.model(input_ids, images=image_tensor, use_cache=False)
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        p_yes = sum(probs[i].item() for i in self.yes_ids)
        p_no = sum(probs[i].item() for i in self.no_ids)
        return p_yes / max(p_yes + p_no, 1e-9)

    # ------------------------------------------------------------------
    class _SentenceStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_len):
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, output_ids, scores, **kwargs):
            new = output_ids[0, self.prompt_len:]
            if new.numel() < 4:
                return False
            text = self.tokenizer.decode(new, skip_special_tokens=True)
            return bool(re.search(r"[.!?]", text))

    @torch.inference_mode()
    def generate_sentence(self, image_tensor, assistant_prefix,
                          max_chunk_tokens=90):
        prompt = self._build_prompt(CAPTION_PROMPT, assistant_prefix)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).to(self.device)
        stop = StoppingCriteriaList(
            [self._SentenceStop(self.tokenizer, input_ids.shape[1])])
        out = self.model.generate(
            input_ids, images=image_tensor, do_sample=False,
            max_new_tokens=max_chunk_tokens, use_cache=True,
            stopping_criteria=stop)
        # official LLaVA returns [input | new]; slice robustly
        if out.shape[1] > input_ids.shape[1] and torch.equal(
                out[0, :4], input_ids[0, :4]):
            new_ids = out[0, input_ids.shape[1]:]
        else:
            new_ids = out[0]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        hit_eos = bool((new_ids == self.tokenizer.eos_token_id).any())
        return text, hit_eos


# ---------------------------------------------------------------------------
# The MSRC caption loop
# ---------------------------------------------------------------------------
def msrc_caption(mm: MSRCModel, image_path, mode="msrc", tau=0.7,
                 max_new_tokens=512, max_sentences=12, log_all_rungs=False):
    """
    mode: 'vanilla'  -> no repair (baseline; with log_all_rungs=True this is
                        the calibration run: all rung scores logged)
          'deletion' -> failed claims are deleted (ConfLVLM-style ablation)
          'msrc'     -> failed claims descend the ladder; delete only if no
                        rung passes (the method)
    Returns (caption, claim_log) where claim_log is a list of dicts.
    """
    image_tensor = mm._prep_image(image_path)
    running, claim_log, total_tokens, sent_idx = "", [], 0, 0

    while total_tokens < max_new_tokens and sent_idx < max_sentences:
        chunk, hit_eos = mm.generate_sentence(image_tensor, running or None)
        if not chunk:
            break
        total_tokens += len(mm.tokenizer(chunk).input_ids)
        sents = split_sentences(chunk)
        if not sents:
            break
        sentence = sents[0]
        sent_idx += 1

        claims = extract_claims(sentence)
        edits, dropped_objects = [], set()

        for c in claims:
            # if the parent object was deleted, child claims die with it
            if c.ctype in ("attribute", "count") and c.head in dropped_objects:
                continue
            record = {"sentence": sentence, "ctype": c.ctype, "head": c.head,
                      "adj": c.adj, "count": c.count,
                      "ladder": list(c.ladder), "scores": {}, "emitted": 0}
            K = n_rungs(c)
            emitted = 0
            if mode == "vanilla":
                if log_all_rungs:
                    for r in range(0, (K + 1) if c.ctype == "object" else 1):
                        record["scores"][str(r)] = mm.verify(
                            question_for(c, r), image_tensor)
            else:
                v0 = mm.verify(question_for(c, 0), image_tensor)
                record["scores"]["0"] = v0
                if v0 >= tau:
                    emitted = 0
                elif mode == "deletion" or c.ctype in ("attribute", "count"):
                    # attribute/count ladder = drop the modifier (rung 1)
                    emitted = 1 if c.ctype in ("attribute", "count") else -1
                else:  # msrc: climb the ladder lazily
                    emitted = -1
                    for r in range(1, K + 1):
                        vr = mm.verify(question_for(c, r), image_tensor)
                        record["scores"][str(r)] = vr
                        if vr >= tau:
                            emitted = r
                            break
                if emitted != 0:
                    e = repair_edit_for(c, emitted)
                    if e is not None:
                        edits.append(e)
                    if c.ctype == "object" and emitted == -1:
                        dropped_objects.add(c.head)
            record["emitted"] = emitted
            claim_log.append(record)

        repaired = apply_repairs(sentence, edits) if edits else sentence
        # a sentence gutted to (nearly) nothing is skipped entirely
        if len(repaired.split()) >= 3:
            running = (running + " " + repaired).strip()
        if hit_eos:
            break

    return running, claim_log
