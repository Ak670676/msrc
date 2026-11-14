# -*- coding: utf-8 -*-
"""
MSRC lattice module: claim extraction + specificity lattices + sentence splicing.

Design notes (fixes over the old token-level UAVCD):
  * All linguistic analysis runs on decoded WORDS, never subwords.
  * Meta-language filter removes "image/photo/picture/..." so the pipeline
    never wastes verification (or, previously, budget) on non-visual nouns.
  * Ladders are ordered by logical ENTAILMENT (golden retriever -> dog ->
    animal): weakening can never create a hallucination (monotone falsity),
    which is the property the conformal risk control theorem needs.
  * Splicing is done with character-span edits on the sentence text, so
    multi-word hypernyms are allowed (no single-token constraint).

Extraction backends: spaCy (preferred, dependency-based) with an NLTK
chunk-grammar fallback so the pipeline runs even without spaCy.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from nltk.corpus import wordnet as wn

try:
    import inflect
    _INFLECT = inflect.engine()
except Exception:
    _INFLECT = None

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
_NLP = None
_BACKEND = None


def _init_backend():
    global _NLP, _BACKEND
    if _BACKEND is not None:
        return
    try:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
        _BACKEND = "spacy"
    except Exception:
        import nltk  # noqa: F401  (data assumed downloaded by setup)
        _BACKEND = "nltk"
    print(f"[lattice] extraction backend: {_BACKEND}")


# ---------------------------------------------------------------------------
# Filters and blocklists
# ---------------------------------------------------------------------------
# Nouns that refer to the picture itself, not the scene. THE critical filter.
META_NOUNS = {
    "image", "photo", "photograph", "picture", "scene", "view", "shot",
    "close-up", "closeup", "foreground", "background", "frame", "camera",
    "angle", "photography", "snapshot", "portrait", "side", "left", "right",
    "top", "bottom", "center", "middle", "front", "back", "part", "area",
    "atmosphere", "setting", "moment", "time", "day", "variety", "type",
    "kind", "lot", "number", "group", "couple", "pair", "bunch", "row",
    "line",
}

# Hypernyms too abstract to be useful (or that verify vacuously).
ABSTRACT_HYPERNYMS = {
    "entity", "physical_entity", "object", "whole", "thing", "unit",
    "artifact", "artefact", "matter", "causal_agent", "living_thing",
    "organism", "abstraction", "attribute", "substance", "instrumentality",
    "instrumentation", "device", "structure", "construction", "creation",
    "article", "item", "part", "body_part", "organ", "natural_object",
    "geological_formation", "system", "being", "agent", "implement",
    "young", "protection", "offspring", "worker", "traveler", "traveller",
    "commodity", "consumer_goods", "covering", "conveyance", "equipment",
}

# WordNet lexnames that indicate a non-visual noun sense.
NONVISUAL_LEXNAMES = {
    "noun.cognition", "noun.communication", "noun.time", "noun.feeling",
    "noun.attribute", "noun.relation", "noun.quantity", "noun.state",
    "noun.act", "noun.motive", "noun.process",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def is_meta_or_nonvisual(lemma: str) -> bool:
    lemma = lemma.lower()
    if lemma in META_NOUNS:
        return True
    syns = wn.synsets(lemma, pos=wn.NOUN)
    if not syns:
        # unknown noun: keep it (could be a real object like "frisbee")
        return False
    # If the dominant sense is non-visual, drop it.
    return syns[0].lexname() in NONVISUAL_LEXNAMES


# ---------------------------------------------------------------------------
# Claim dataclass
# ---------------------------------------------------------------------------
@dataclass
class Claim:
    ctype: str                 # 'object' | 'attribute' | 'count'
    head: str                  # head noun lemma (singular)
    head_surface: str          # surface form of the head noun
    plural: bool
    span: Tuple[int, int]      # char span of the FULL noun phrase in sentence
    head_span: Tuple[int, int] # char span of the head noun token
    adj: Optional[str] = None
    adj_span: Optional[Tuple[int, int]] = None
    count: Optional[int] = None
    count_span: Optional[Tuple[int, int]] = None
    ladder: List[str] = field(default_factory=list)  # object hypernym rungs


# ---------------------------------------------------------------------------
# Lemmatization / pluralization helpers
# ---------------------------------------------------------------------------
_KEEP_SINGULAR = {"bus", "boss", "gas", "lens", "dress", "grass", "glass",
                  "canvas", "tennis", "cactus", "octopus", "walrus", "iris"}


def singularize(word: str) -> str:
    # FIX (unified): inflect maps 'bus'->'bu', 'tennis'->'tenni',
    # 'grass'->'gras'. 'bus' is a COCO category, so bus claims were silently
    # dropped from the guarantee vocabulary. Keep the word when it is already
    # a singular noun and inflect's output is not a WordNet noun.
    if word.lower() in _KEEP_SINGULAR:
        return word.lower()
    if _INFLECT is not None:
        s = _INFLECT.singular_noun(word)
        if s:
            try:
                from nltk.corpus import wordnet as _wn
                if (not _wn.synsets(s, pos=_wn.NOUN)
                        and _wn.synsets(word, pos=_wn.NOUN)):
                    return word.lower()
            except Exception:
                pass
            return s
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def pluralize(word: str) -> str:
    if _INFLECT is not None:
        p = _INFLECT.plural_noun(word)
        if p:
            return p
    w = word
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    return w + "s"


# ---------------------------------------------------------------------------
# Hypernym ladder (the specificity lattice for object claims)
# ---------------------------------------------------------------------------
# Caption-natural generalizations. If one of these appears anywhere on the
# upward path it is always kept; other rungs only fill remaining slots.
PREFERRED_RUNGS = {
    "animal", "person", "bird", "fish", "insect", "plant", "tree", "flower",
    "vehicle", "food", "fruit", "vegetable", "furniture", "container",
    "tool", "toy", "clothing", "building", "room", "appliance", "machine",
    "boat", "aircraft", "dog", "cat", "sign", "light", "bag",
}


def hypernym_ladder(lemma: str, max_depth: int = 3) -> List[str]:
    """Entailment-ordered chain of single-word hypernyms, preferring
    caption-natural rungs. E.g. 'puppy' -> ['dog', 'animal'].
    Order along the chain preserves specificity (nearest = most specific),
    so emitting a later rung is a strict logical weakening.
    """
    lemma = lemma.lower().replace(" ", "_")
    syns = wn.synsets(lemma, pos=wn.NOUN)
    if not syns:
        return []
    candidates, seen, node = [], {lemma}, syns[0]

    def lemma_of(syn):
        best = None
        for lm in syn.lemmas():
            name = lm.name().lower()
            if ("_" in name or name in seen or name in ABSTRACT_HYPERNYMS
                    or name in META_NOUNS or len(name) <= 2):
                continue
            if name in ("animal", "person", "vehicle", "food", "plant",
                        "furniture", "bird"):
                return name, 3
            if name in PREFERRED_RUNGS:
                return name, 2
            if best is None:
                best = name
        return (best, 1) if best else (None, 0)

    for _ in range(10):                      # walk far enough to reach 'animal'
        hypers = node.hypernyms() or node.instance_hypernyms()
        if not hypers:
            break
        # prefer the branch whose lemma is caption-natural (fixes
        # wheeled_vehicle -> container vs. vehicle)
        scored = sorted(((lemma_of(h), h) for h in hypers),
                        key=lambda x: -x[0][1])
        (name, score), node = scored[0]
        if name is not None:
            candidates.append(name)
            seen.add(name)
    preferred = [c for c in candidates if c in PREFERRED_RUNGS]
    others = [c for c in candidates if c not in PREFERRED_RUNGS]
    ladder = []
    for c in candidates:                     # keep chain order (entailment!)
        if c in preferred or (c in others[:max(0, max_depth - len(preferred))]):
            ladder.append(c)
        if len(ladder) >= max_depth:
            break
    return ladder


def is_ancestor(hyper_lemma: str, hypo_lemma: str) -> bool:
    """True iff hyper_lemma is a WordNet hypernym (ancestor) of hypo_lemma,
    or the same lemma. Used to label emitted hypernyms and to compute SWC."""
    hyper_lemma = hyper_lemma.lower().replace(" ", "_")
    hypo_lemma = hypo_lemma.lower().replace(" ", "_")
    if hyper_lemma == hypo_lemma:
        return True
    hypo_syns = wn.synsets(hypo_lemma, pos=wn.NOUN)
    hyper_syns = set(wn.synsets(hyper_lemma, pos=wn.NOUN))
    if not hypo_syns or not hyper_syns:
        return False
    for s in hypo_syns[:3]:
        # FIX (unified): some WordNet entries in this nltk build yield None
        # hypernyms, which crash closure(); drop them.
        try:
            closure = set(s.closure(lambda x: [h for h in x.hypernyms() +
                                               x.instance_hypernyms()
                                               if h is not None]))
        except AttributeError:
            continue
        if hyper_syns & closure:
            return True
    return False


def wn_depth(lemma: str) -> int:
    syns = wn.synsets(lemma.lower().replace(" ", "_"), pos=wn.NOUN)
    return syns[0].min_depth() if syns else 0


# ---------------------------------------------------------------------------
# Extraction — spaCy backend
# ---------------------------------------------------------------------------
def _extract_spacy(sentence: str) -> List[Claim]:
    doc = _NLP(sentence)
    claims = []
    for chunk in doc.noun_chunks:
        head = chunk.root
        if head.pos_ not in ("NOUN", "PROPN"):
            continue
        lemma = singularize(head.text)
        if is_meta_or_nonvisual(lemma):
            continue
        plural = head.tag_ in ("NNS", "NNPS")
        c = Claim(
            ctype="object", head=lemma, head_surface=head.text, plural=plural,
            span=(chunk.start_char, chunk.end_char),
            head_span=(head.idx, head.idx + len(head.text)),
            ladder=hypernym_ladder(lemma),
        )
        claims.append(c)
        for child in head.children:
            if child.dep_ == "amod" and child.pos_ == "ADJ":
                claims.append(Claim(
                    ctype="attribute", head=lemma, head_surface=head.text,
                    plural=plural, span=(chunk.start_char, chunk.end_char),
                    head_span=(head.idx, head.idx + len(head.text)),
                    adj=child.text.lower(),
                    adj_span=(child.idx, child.idx + len(child.text)),
                ))
            elif child.dep_ == "nummod" or (child.like_num and child.dep_ == "det"):
                num = NUMBER_WORDS.get(child.text.lower())
                if num is None:
                    try:
                        num = int(child.text)
                    except ValueError:
                        continue
                claims.append(Claim(
                    ctype="count", head=lemma, head_surface=head.text,
                    plural=plural, span=(chunk.start_char, chunk.end_char),
                    head_span=(head.idx, head.idx + len(head.text)),
                    count=num,
                    count_span=(child.idx, child.idx + len(child.text)),
                ))
    return claims


# ---------------------------------------------------------------------------
# Extraction — NLTK fallback (chunk grammar: DT? CD? JJ* NN+)
# ---------------------------------------------------------------------------
def _extract_nltk(sentence: str) -> List[Claim]:
    from nltk import pos_tag, RegexpParser
    from nltk.tokenize import TreebankWordTokenizer

    tk = TreebankWordTokenizer()
    spans = list(tk.span_tokenize(sentence))
    words = [sentence[a:b] for a, b in spans]
    tagged = pos_tag(words)
    grammar = r"NP: {<DT|PRP\$>?<CD>?<JJ|JJR|JJS>*<NN|NNS|NNP|NNPS>+}"
    tree = RegexpParser(grammar).parse(tagged)

    claims, idx = [], 0
    for node in tree:
        if not hasattr(node, "label"):
            idx += 1
            continue
        leaves = node.leaves()
        n = len(leaves)
        chunk_spans = spans[idx:idx + n]
        chunk_tags = [t for _, t in leaves]
        chunk_words = [w for w, _ in leaves]
        idx += n

        # head noun = last NN* token
        head_i = max(i for i, t in enumerate(chunk_tags) if t.startswith("NN"))
        head_surface = chunk_words[head_i]
        lemma = singularize(head_surface)
        if is_meta_or_nonvisual(lemma):
            continue
        plural = chunk_tags[head_i] in ("NNS", "NNPS")
        span = (chunk_spans[0][0], chunk_spans[-1][1])
        head_span = chunk_spans[head_i]

        claims.append(Claim(
            ctype="object", head=lemma, head_surface=head_surface,
            plural=plural, span=span, head_span=head_span,
            ladder=hypernym_ladder(lemma),
        ))
        for i, t in enumerate(chunk_tags):
            if t.startswith("JJ") and i < head_i:
                claims.append(Claim(
                    ctype="attribute", head=lemma, head_surface=head_surface,
                    plural=plural, span=span, head_span=head_span,
                    adj=chunk_words[i].lower(), adj_span=chunk_spans[i],
                ))
            elif t == "CD" and i < head_i:
                num = NUMBER_WORDS.get(chunk_words[i].lower())
                if num is None:
                    try:
                        num = int(chunk_words[i])
                    except ValueError:
                        continue
                claims.append(Claim(
                    ctype="count", head=lemma, head_surface=head_surface,
                    plural=plural, span=span, head_span=head_span,
                    count=num, count_span=chunk_spans[i],
                ))
    return claims


def extract_claims(sentence: str) -> List[Claim]:
    _init_backend()
    try:
        if _BACKEND == "spacy":
            return _extract_spacy(sentence)
        return _extract_nltk(sentence)
    except Exception as e:
        print(f"[lattice] extraction failed on: {sentence[:60]!r} ({e})")
        return []


# ---------------------------------------------------------------------------
# Splicing: apply repairs as character-span edits, then clean up grammar
# ---------------------------------------------------------------------------
def _fix_articles(text: str) -> str:
    def sub(m):
        art, word = m.group(1), m.group(2)
        vowel = word[0].lower() in "aeiou"
        new = ("an" if vowel else "a")
        if art[0].isupper():
            new = new.capitalize()
        return f"{new} {word}"
    return re.sub(r"\b([Aa]n?)\s+(\w+)", sub, text)


def _cleanup(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"(,\s*)+,", ",", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\b(and|with|of|on|in|at|near|beside|by)\s*([,.])", r"\2", text)
    text = re.sub(r"^\s*(and|with|,)\s+", "", text, flags=re.I)
    text = re.sub(r"\s*,\s*and\s*\.", ".", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = _fix_articles(text)
    text = text.strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


def _cleanup_strict(text: str) -> str:
    """Whitespace only: Eq. (4) removes sigma and nothing else, so no word
    outside an edited span may be added or removed here."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text.strip()


def apply_repairs(sentence: str, edits: List[Tuple[int, int, str]],
                  strict: bool = False) -> str:
    """edits: list of (start, end, replacement). Deletions inside a deleted
    outer span are dropped; remaining edits applied right-to-left.

    strict=True is Eq. (4) exactly: each deletion removes its span sigma and
    nothing else (no preposition swallowing, no article or punctuation
    rewriting beyond whitespace)."""
    # Drop edits nested inside a full-span deletion
    deletions = [(s, e) for (s, e, r) in edits if r == ""]
    kept = []
    for (s, e, r) in edits:
        nested = any(ds <= s and e <= de and (s, e) != (ds, de)
                     for (ds, de) in deletions)
        if not nested:
            kept.append((s, e, r))
    kept.sort(key=lambda x: x[0], reverse=True)
    out = sentence
    PRE_FUNC = re.compile(
        r"(?:\b(?:and\s+|while\s+)?(?:holding|carrying|wearing|using|eating|"
        r"drinking|riding|reading|watching|touching|grabbing)\s+"
        r"|\b(?:with|and|of|on|in|at|near|beside|by|next to|under|over|"
        r"behind|around|,)\s+)$", re.I)
    for (s, e, r) in kept:
        if r == "" and not strict:
            # swallow a preceding preposition/conjunction ("with", "and", ",")
            m = PRE_FUNC.search(out[:s])
            if m:
                s = m.start()
            if s > 0 and out[s - 1] == " ":
                s -= 1
        out = out[:s] + r + out[e:]
    return _cleanup_strict(out) if strict else _cleanup(out)


def repair_edit_for(claim: Claim, emitted_rung: int) -> Optional[Tuple[int, int, str]]:
    """Translate an emission decision into a character edit.
    emitted_rung: 0 = keep as is (no edit) ... k = k-th weakening; -1 = delete.
    """
    if emitted_rung == 0:
        return None
    if claim.ctype == "object":
        if emitted_rung == -1:
            return (claim.span[0], claim.span[1], "")
        hyper = claim.ladder[emitted_rung - 1]
        surface = pluralize(hyper) if claim.plural else hyper
        return (claim.head_span[0], claim.head_span[1], surface)
    if claim.ctype == "attribute":
        # only weakening: drop the adjective
        return (claim.adj_span[0], claim.adj_span[1], "")
    if claim.ctype == "count":
        # rung 1: drop the number -> bare plural ("dogs")
        return (claim.count_span[0], claim.count_span[1], "")
    return None


# ---------------------------------------------------------------------------
# Verification questions per rung
# ---------------------------------------------------------------------------
def question_for(claim: Claim, rung: int) -> str:
    if claim.ctype == "object":
        noun = claim.head if rung == 0 else claim.ladder[rung - 1]
        art = "an" if noun[0].lower() in "aeiou" else "a"
        return f"Is there {art} {noun} in the image?"
    if claim.ctype == "attribute":
        return f"Is the {claim.head} {claim.adj}?"
    if claim.ctype == "count":
        return (f"Are there exactly {claim.count} "
                f"{pluralize(claim.head)} in the image?")
    raise ValueError(claim.ctype)


def n_rungs(claim: Claim) -> int:
    """Number of weakening rungs ABOVE rung 0 (excluding deletion)."""
    if claim.ctype == "object":
        return len(claim.ladder)
    return 1 if claim.ctype in ("attribute", "count") else 0


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
