# -*- coding: utf-8 -*-
"""Grammar-preserving deletion (so1W W2 / Q1 ablation).

Eq. (4) removes the noun phrase sigma and nothing else, which can leave the
sentence ungrammatical ("placed on the, ..."). This variant removes the
smallest SYNTACTIC unit that contains sigma and whose removal leaves a
well-formed remainder, read off the dependency parse of the original
sentence (spaCy en_core_web_sm, the claim extractor's parser):

  sigma is a conjunct            -> sigma + its coordinator / comma
                                    ("a sink and a mirror" -> "a mirror")
  sigma is the object of a prep. -> the whole prepositional phrase
                                    ("placed on the sink" -> "placed")
  sigma is a core argument       -> the smallest clause containing it
    (subject, object, attribute,    (relative / adverbial / complement /
     existential "there is X")      coordinated clause) with its comma or
                                    coordinator; if that clause is the main
                                    clause, the whole sentence is dropped
  anything else                  -> sigma only (Eq. 4)

Substitutions are applied unchanged. Only deletions are expanded, so every
claim the policy kept stays in the text unless it sits inside a removed
clause; those are counted by the caller as collateral removals.
"""
import re

_NLP = [None]
ARG = {"nsubj", "nsubjpass", "dobj", "attr", "oprd", "dative", "csubj",
       "agent", "expl", "appos"}
CLAUSE = {"relcl", "advcl", "acl", "ccomp", "xcomp", "conj", "parataxis"}


def nlp():
    if _NLP[0] is None:
        import spacy
        _NLP[0] = spacy.load("en_core_web_sm")
    return _NLP[0]


def _span(tok):
    sub = list(tok.subtree)
    return sub[0].idx, sub[-1].idx + len(sub[-1].text)


def _own_span(tok):
    """subtree of tok WITHOUT its own coordinated siblings (conj / cc and the
    commas between them), i.e. just this conjunct"""
    drop = set()
    for c in tok.children:
        if c.dep_ in {"conj", "cc"} or (c.dep_ == "punct" and c.i > tok.i):
            drop |= {t.i for t in c.subtree}
    toks = [t for t in tok.subtree if t.i not in drop]
    return toks[0].idx, toks[-1].idx + len(toks[-1].text)


def _with_left_glue(doc, a, b):
    """extend a span left over a coordinator / comma directly before it"""
    toks = [t for t in doc if t.idx + len(t.text) <= a]
    while toks and (toks[-1].dep_ == "cc" or toks[-1].text in {",", ";"}):
        a = toks[-1].idx
        toks.pop()
    return a, b


def _with_right_glue(doc, a, b):
    toks = [t for t in doc if t.idx >= b]
    while toks and (toks[0].dep_ == "cc" or toks[0].text in {",", ";"}):
        b = toks[0].idx + len(toks[0].text)
        toks.pop(0)
    return a, b


def expand(doc, s, e):
    """-> (start, end) of the unit to delete for the deletion span [s, e),
    or None to drop the whole sentence."""
    inside = [t for t in doc if s <= t.idx < e]
    if not inside:
        return s, e
    head = next((t for t in inside if not (s <= t.head.idx < e)
                 or t.head == t), inside[-1])
    dep = head.dep_
    if dep == "conj":
        return _with_left_glue(doc, *_own_span(head))
    if any(c.dep_ == "conj" for c in head.children):
        # first conjunct: remove it and the glue up to the next conjunct
        nxt = min(c for c in head.children if c.dep_ == "conj")
        a = _own_span(head)[0]
        return a, next(nxt.subtree).idx
    if dep == "pobj":
        prep = head.head
        # compound prepositions: "next to X", "in front of X", "on top of X"
        while (prep.head != prep and prep.head.dep_ in {"prep", "advmod", "pobj", "npadvmod"}
               and prep.head.pos_ in {"ADV", "ADP", "NOUN", "ADJ"}
               and prep.head.i < prep.i and prep.i - prep.head.i <= 2):
            prep = prep.head
        a, b = _span(prep)
        if prep.dep_ in {"prep", "agent", "dative"}:
            return _with_left_glue(doc, a, b)
        return a, b
    if dep in ARG or (dep == "ROOT" and head.pos_ in {"NOUN", "PROPN"}):
        v = head.head if dep != "ROOT" else head
        while v.dep_ not in CLAUSE and v.dep_ != "ROOT" and v.head != v:
            v = v.head
        if v.dep_ == "ROOT" or v.head == v:
            return None                       # main clause: drop sentence
        a, b = _span(v)
        if v.dep_ == "conj":
            return _with_left_glue(doc, a, b)
        return _with_left_glue(doc, a, b)
    return s, e


def apply_grammar(sentence, edits):
    """edits: [(start, end, replacement)]. -> new sentence ('' = dropped)."""
    doc = nlp()(sentence)
    spans, subs = [], []
    for s, e, r in edits:
        if r:
            subs.append((s, e, r))
            continue
        x = expand(doc, s, e)
        if x is None:
            return ""
        spans.append(x)
    # union of deletion spans
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    # substitutions inside a deleted span are dropped with it
    subs = [x for x in subs if not any(a <= x[0] and x[1] <= b for a, b in merged)]
    ops = [(a, b, "") for a, b in merged] + subs
    out = sentence
    for a, b, r in sorted(ops, key=lambda x: -x[0]):
        out = out[:a] + r + out[b:]
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    out = re.sub(r"([,;])\s*([,.;!?])", r"\2", out)
    out = re.sub(r"^\s*[,;]\s*", "", out).strip()
    # two remaining conjuncts need no serial comma: "a car, and a bicycle"
    out = re.sub(r"^(.*?\b(?:a|an|the|two|some)\s+\w+), (and|or) ((?:a|an|the)\s+\w+[.]?)$", r"\1 \2 \3", out)
    out = re.sub(r"\b([Aa])n? (\w)", lambda m: (m.group(1) + ("n" if m.group(2).lower() in "aeiou" else "")) + " " + m.group(2), out)
    if not re.search(r"[A-Za-z]", out):
        return ""
    if out and out[-1] not in ".!?":
        out += "."
    return out[0].upper() + out[1:]
