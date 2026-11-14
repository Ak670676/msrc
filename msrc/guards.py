# -*- coding: utf-8 -*-
"""
MSRC v3 guards — the label-free policy filters that make verified
re-grounding behave as described in the paper.

Everything here is a deterministic function of (caption text, claim, model)
and never touches calibration labels, so folding these into the pipeline
preserves the conformal guarantee (they are part of the deployed score /
policy, and calibration must be rerun with them active).

Three guards:

  G1  SUBSTITUTABILITY  — only concrete, visual object heads may be
      re-grounded. Filler nouns ("addition", "other", "closer", "touch"),
      scene nouns ("room", "background"), and unknown tokens ("focu") are
      not object claims at all: they are skipped entirely (no verification,
      no edit), which is both cheaper and fluency-preserving.

  G2  DUPLICATE  — a substitute that already appears in the caption adds no
      information and produces degenerate text ("a computer, a computer").
      Rejected -> fall back to deletion.

  G3  COMPOUND  — "hot dog" must be verified as "hot dog", not as head
      "dog". Compound COCO categories are verified whole and never
      head-substituted.

Diagnostics from the pre-guard LLaVA run that motivated these:
  73.4% of substitutions repeated a word already in the caption (G2)
  23.5% substituted an abstract/filler head (G1)
  only 3.1% were clean concrete re-groundings
"""
import re
from typing import Optional, Set

from nltk.corpus import wordnet as wn

try:
    from msrc.chair_utils import WORD2CAT, DOUBLE_WORDS
except Exception:                                    # standalone use
    WORD2CAT, DOUBLE_WORDS = {}, {}

# ---------------------------------------------------------------------------
# G1 — substitutability
# ---------------------------------------------------------------------------
# WordNet lexnames that denote a concrete, depictable thing. Whitelist, not
# blacklist: the old blacklist let "addition"(noun.artifact),
# "closer"(noun.person), "touch"(noun.event), "corner"(noun.location) through.
VISUAL_LEXNAMES = {
    "noun.animal", "noun.artifact", "noun.body", "noun.food",
    "noun.plant", "noun.object", "noun.person",
}

# Heads that pass the lexname test but are never a depictable object claim
# in caption language. Measured offenders from out/eval_fixed_recal_claims.
FILLER_HEADS = {
    "addition", "other", "closer", "touch", "focu", "focus", "space",
    "corner", "edge", "side", "total", "detail", "variety", "assortment",
    "array", "arrangement", "collection", "combination", "mix", "sort",
    "manner", "way", "style", "element", "aspect", "presence", "feature",
    "sense", "look", "sight", "glimpse", "view", "distance", "amount",
    "piece", "item", "object", "thing", "stuff", "use", "addition",
}

# Scene / mass nouns: real but not localizable object claims. Substituting
# them ("room" -> "laptop") is always wrong.
SCENE_HEADS = {
    "room", "kitchen", "bathroom", "bedroom", "office", "street", "road",
    "field", "beach", "park", "city", "town", "village", "area", "place",
    "location", "background", "foreground", "sky", "water", "ground",
    "floor", "wall", "ceiling", "surface", "landscape", "scenery", "scene",
    "environment", "atmosphere", "setting", "interior", "exterior",
}

_VIS_CACHE = {}


def is_substitutable_head(head: str) -> bool:
    """G1: may this head be verified and re-grounded at all?

    True  -> real object claim (COCO or open-vocabulary, e.g. 'railing')
    False -> not an object claim; skip it entirely (no verify, no edit)
    """
    h = (head or "").lower().strip()
    if not h or not h.isalpha():
        return False
    if h in _VIS_CACHE:
        return _VIS_CACHE[h]
    ok = True
    if h in FILLER_HEADS or h in SCENE_HEADS:
        ok = False
    elif h in WORD2CAT:                     # COCO vocabulary always wins
        ok = True
    else:
        syns = [x for x in wn.synsets(h, pos=wn.NOUN) if x is not None]
        if not syns:
            # Unknown token ('focu', OCR noise). The old code kept these on
            # the theory they might be rare objects; measurement says they
            # are overwhelmingly noise, and an unverifiable head cannot be
            # safely substituted.
            ok = False
        else:
            ok = syns[0].lexname() in VISUAL_LEXNAMES
    _VIS_CACHE[h] = ok
    return ok


# ---------------------------------------------------------------------------
# G2 — duplicate substitute
# ---------------------------------------------------------------------------
def _appears(word: str, text: str) -> int:
    if not word or not text:
        return 0
    return len(re.findall(r"\b" + re.escape(word.lower()) + r"\w*\b",
                          text.lower()))


def duplicate_blocked(sub: str, sentence: str, running: str,
                      scope: str = "caption") -> bool:
    """G2: True if accepting `sub` would repeat an existing word.

    scope='caption'  (default, strict) block if it occurs anywhere already
    scope='sentence' (lenient)         block only within the same sentence
    """
    if not sub:
        return True
    if _appears(sub, sentence) > 0:
        return True
    if scope == "caption" and _appears(sub, running) > 0:
        return True
    return False


def mentioned_words(sentence: str, running: str, limit: int = 12) -> Set[str]:
    """Words to tell the re-grounding prompt to avoid (prompt-level G2)."""
    toks = re.findall(r"[a-z]{3,}", (running + " " + sentence).lower())
    seen, out = set(), []
    for t in toks:
        if t in WORD2CAT and t not in seen:
            seen.add(t)
            out.append(t)
    return set(out[-limit:])


# ---------------------------------------------------------------------------
# G3 — compound COCO categories
# ---------------------------------------------------------------------------
def compound_at(sentence: str, claim) -> Optional[str]:
    """G3: if the head is the tail of a COCO two-word category present in
    the text ('hot dog', 'teddy bear', 'cell phone', 'stop sign', 'tennis
    racket', 'wine glass', ...), return that compound string.

    Such claims are verified as the compound and never head-substituted.
    """
    if not DOUBLE_WORDS:
        return None
    s, e = claim.head_span
    prefix = sentence[:s].rstrip()
    if not prefix:
        return None
    prev = re.findall(r"[A-Za-z]+$", prefix)
    if not prev:
        return None
    bigram = f"{prev[0].lower()} {claim.head.lower()}"
    return bigram if bigram in DOUBLE_WORDS else None


def question_for_compound(compound: str) -> str:
    art = "an" if compound[0].lower() in "aeiou" else "a"
    return f"Is there {art} {compound} in the image?"


# ---------------------------------------------------------------------------
# Positional re-grounding prompt (replaces the salient-object prompt)
# ---------------------------------------------------------------------------
def reground_prompt(head: str, np_text: str, sentence: str,
                    avoid: Optional[Set[str]] = None) -> str:
    """Ask about the object at THAT position, not the most salient object.

    The old prompt ("What similar object is clearly visible instead?")
    returned the globally salient object, which is why 73% of substitutes
    repeated a word already in the caption.
    """
    q = (f'A description of this image says: "{sentence.strip()}" '
         f'The phrase "{np_text.strip()}" may be wrong. '
         f"Looking only at that part of the image, what object is actually "
         f"there? Answer with one common noun.")
    if avoid:
        q += " Do not answer with: " + ", ".join(sorted(avoid)) + "."
    return q


# ---------------------------------------------------------------------------
# Decision helper used by both loops
# ---------------------------------------------------------------------------
def accept_substitute(sub, v_sub, tau_sub, sentence, running,
                      dup_scope="caption", semantic_ok=True,
                      semantic_reason=None):
    """Return (accepted: bool, reason: str).

    semantic_ok: result of semantic_substitute_compatible(). A HARD guard --
    it is a deterministic function of the two words, not a score, so it is
    not calibrated via tau_sub. It can only REJECT a substitute (moving the
    claim to deletion, which carries zero risk), so adding it cannot raise
    the certified bound.
    """
    if not sub:
        return False, "no_candidate"
    if not is_substitutable_head(sub):
        return False, "sub_not_visual"
    if duplicate_blocked(sub, sentence, running, dup_scope):
        return False, "duplicate"
    if not semantic_ok:
        return False, semantic_reason or "semantic_incompatible"
    if v_sub < tau_sub:
        return False, "below_tau_sub"
    return True, "accepted"


# ---------------------------------------------------------------------------
# MSRC-v3 deterministic semantic substitute compatibility
# ---------------------------------------------------------------------------

_SEM_SENSE_CACHE = {}


def _sense_family(syn):
    """Map a WordNet noun sense to a coarse physical semantic family.

    Unlike the extraction visuality filter, this also recovers noun.Tops
    concepts such as person.n.01 by inspecting the synset/hypernym lemmas.
    """
    if syn is None:
        return None

    lex = syn.lexname()

    if lex == "noun.person":
        return "person"
    if lex == "noun.animal":
        return "animal"
    if lex == "noun.plant":
        return "plant"
    if lex == "noun.food":
        return "food"
    if lex == "noun.body":
        return "body"
    if lex == "noun.artifact":
        return "artifact"
    if lex == "noun.object":
        return "object"

    # Some core physical concepts live in noun.Tops rather than one of the
    # standard visual lexnames.  person.n.01 is the important example.
    try:
        nodes = {syn}
        nodes.update(
            syn.closure(
                lambda x: x.hypernyms() + x.instance_hypernyms()
            )
        )
    except Exception:
        nodes = {syn}

    names = set()
    for node in nodes:
        try:
            names.update(
                lm.name().lower().replace("_", " ")
                for lm in node.lemmas()
            )
        except Exception:
            pass

    if "person" in names or "individual" in names:
        return "person"

    return None


def _semantic_senses(word, limit=8):
    """Ranked physical WordNet senses with their coarse semantic families."""
    w = (word or "").strip().lower().replace(" ", "_")

    if not w:
        return []

    key = (w, int(limit))
    if key in _SEM_SENSE_CACHE:
        return _SEM_SENSE_CACHE[key]

    try:
        syns = wn.synsets(w, pos=wn.NOUN)
    except LookupError:
        syns = []

    out = []

    for rank, syn in enumerate(syns[:limit]):
        fam = _sense_family(syn)

        if fam is not None:
            out.append((rank, syn, fam))

    _SEM_SENSE_CACHE[key] = out
    return out


def _multisense_taxonomic_family(a_senses, b_senses):
    """Return a shared semantic family containing a direct taxonomic match.

    This handles polysemous nouns such as cake, whose dominant physical sense
    may differ from the sense relevant to cupcake.

    Crucially, the matching senses must belong to the SAME coarse semantic
    family, so unrelated rare senses cannot bridge arbitrary substitutions.
    """
    for _, a, afam in a_senses:
        for _, b, bfam in b_senses:
            if afam != bfam:
                continue

            if (
                a == b
                or _syn_is_ancestor(a, b)
                or _syn_is_ancestor(b, a)
            ):
                return afam

    return None


def _dominant_semantic_family(senses):
    """Family of the earliest plausible physical sense."""
    if not senses:
        return None

    rank, syn, fam = min(senses, key=lambda x: x[0])
    return fam


def _family_senses(senses, family):
    return [
        syn
        for _, syn, fam in senses
        if fam == family
    ]


def _syn_is_ancestor(ancestor, descendant):
    if ancestor == descendant:
        return True

    try:
        closure = set(
            descendant.closure(
                lambda x: x.hypernyms() + x.instance_hypernyms()
            )
        )
    except Exception:
        closure = set()

    return ancestor in closure


def _taxonomically_related(a_senses, b_senses):
    """Synonym/hypernym/hyponym relation within relevant semantic senses."""
    for a in a_senses:
        for b in b_senses:
            if (
                a == b
                or _syn_is_ancestor(a, b)
                or _syn_is_ancestor(b, a)
            ):
                return True

    return False


def _holonym_closure(syn, max_depth=3):
    seen = set()
    frontier = [(syn, 0)]

    while frontier:
        cur, depth = frontier.pop(0)

        if depth >= max_depth:
            continue

        try:
            nxt = (
                cur.part_holonyms()
                + cur.member_holonyms()
                + cur.substance_holonyms()
            )
        except Exception:
            nxt = []

        for h in nxt:
            if h is None or h in seen:
                continue

            seen.add(h)
            frontier.append((h, depth + 1))

    seen.discard(None)
    return seen


def _part_whole_related(a_senses, b_senses):
    """Part/whole only within the selected dominant semantic family."""
    # FIX (unified): sense lists can carry None; nltk Synset.__eq__ then
    # raises AttributeError. A missing sense relates to nothing.
    a_senses = [x for x in a_senses if x is not None]
    b_senses = [x for x in b_senses if x is not None]

    for a in a_senses:
        ah = _holonym_closure(a)

        for b in b_senses:
            if any(
                h == b or _syn_is_ancestor(h, b)
                for h in ah
            ):
                return True

            bh = _holonym_closure(b)

            if any(
                h == a or _syn_is_ancestor(h, a)
                for h in bh
            ):
                return True

    return False


def _best_wup(a_senses, b_senses):
    best = 0.0

    for a in a_senses:
        for b in b_senses:
            try:
                score = a.wup_similarity(b)
            except Exception:
                score = None

            if score is not None:
                best = max(best, float(score))

    return best


def _preferred_semantic_families(senses):
    """Caption-natural hypernym families for compatible WordNet senses."""
    try:
        from msrc.lattice import PREFERRED_RUNGS
    except Exception:
        PREFERRED_RUNGS = set()

    fam = set()

    for syn in senses:
        nodes = {syn}

        try:
            nodes.update(
                syn.closure(
                    lambda x: x.hypernyms() + x.instance_hypernyms()
                )
            )
        except Exception:
            pass

        for node in nodes:
            try:
                lemmas = node.lemmas()
            except Exception:
                lemmas = []

            for lemma in lemmas:
                name = lemma.name().lower().replace("_", " ")

                if name in PREFERRED_RUNGS:
                    fam.add(name)

    return fam


def _question_mentions_head(question, head):
    """Morphology-aware check for the original noun in the question."""
    try:
        from msrc.lattice import singularize
    except Exception:
        singularize = lambda x: x.lower()

    q = (question or "").lower()
    h = (head or "").lower().strip()

    if not q or not h:
        return False

    tokens = re.findall(r"[a-z][a-z'-]*", q)

    forms = set()

    for tok in tokens:
        forms.add(tok)

        try:
            forms.add(singularize(tok))
        except Exception:
            pass

    compact_h = re.sub(r"[^a-z0-9]", "", h)

    compact_forms = {
        re.sub(r"[^a-z0-9]", "", x)
        for x in forms
    }

    # e.g. i-phone / i phone -> iphone
    for i in range(len(tokens) - 1):
        compact_forms.add(
            re.sub(
                r"[^a-z0-9]",
                "",
                tokens[i] + tokens[i + 1],
            )
        )

    return h in forms or compact_h in compact_forms


def semantic_substitute_compatible(head, sub, question=""):
    """Deterministic same-role guard for MSRC re-grounding.

    Key difference from extraction:
      extraction may preserve ANY plausible visual sense;
      substitution must compare the DOMINANT physical semantic role.

    Returns:
        (ok, reason, metadata)
    """
    h = (head or "").strip().lower()
    s = (sub or "").strip().lower()

    meta = {
        "head": h,
        "sub": s,
        "question_mentions_head": False,
        "head_family": None,
        "sub_family": None,
        "head_lexnames": [],
        "sub_lexnames": [],
        "head_families": [],
        "sub_families": [],
        "wup": 0.0,
    }

    if not h or not s:
        return False, "semantic_missing", meta

    if h == s:
        return True, "semantic_same", meta

    hs_all = _semantic_senses(h)
    ss_all = _semantic_senses(s)

    if not hs_all or not ss_all:
        return False, "semantic_unknown", meta

    hfam = _dominant_semantic_family(hs_all)
    sfam = _dominant_semantic_family(ss_all)

    meta["head_family"] = hfam
    meta["sub_family"] = sfam

    # MSRC-v3 FINAL FREEZE: dominant-family hard semantic gate
    #
    # Cross-dominant-family WordNet rescue is deliberately disabled.
    # DEV auditing showed that rare senses can otherwise bridge unrelated
    # entities (person->chair, person->bear, officer->pig, etc.).
    #
    # Same-family polysemy/taxonomic reasoning below remains available.
    if hfam != sfam:
        return False, "semantic_class_mismatch", meta
    meta["head_lexnames"] = sorted({
        syn.lexname() for _, syn, _ in hs_all
    })
    meta["sub_lexnames"] = sorted({
        syn.lexname() for _, syn, _ in ss_all
    })

    q_mentions = _question_mentions_head(question, h)
    meta["question_mentions_head"] = q_mentions

    # ------------------------------------------------------------------
    # Strong multi-sense taxonomic evidence may rescue a legitimate
    # polysemous pair BEFORE dominant-family rejection.
    #
    # Example:
    #   cake -> cupcake
    # The first physical sense of cake is not necessarily noun.food, but a
    # later food sense is taxonomically related to cupcake.
    #
    # The related senses must share the SAME coarse family, preventing a rare
    # unrelated sense from bridging arbitrary semantic classes.
    # ------------------------------------------------------------------
    tax_family = _multisense_taxonomic_family(hs_all, ss_all)

    if tax_family is not None:
        meta["taxonomic_family"] = tax_family
        return True, "semantic_taxonomic_multisense", meta

    # ------------------------------------------------------------------
    # Otherwise protect against polysemy using dominant physical roles.
    #
    # woman -> chair:
    #   dominant woman = person
    #   dominant chair = artifact
    #
    # The rare chairperson sense must not rescue it.
    # ------------------------------------------------------------------
    if hfam != sfam:
        return False, "semantic_class_mismatch", meta

    hs = _family_senses(hs_all, hfam)
    ss = _family_senses(ss_all, sfam)

    # ------------------------------------------------------------------
    # Taxonomic relation BEFORE meronym/holonym testing.
    #
    # This prevents an unrelated secondary sense from making:
    #   bird -> toucan
    # look like a part/whole relation.
    # ------------------------------------------------------------------
    if _taxonomically_related(hs, ss):
        return True, "semantic_taxonomic", meta

    # Genuine same-family part/whole switches are not same-referent repair.
    if _part_whole_related(hs, ss):
        return False, "semantic_part_whole", meta

    hpf = _preferred_semantic_families(hs)
    spf = _preferred_semantic_families(ss)

    meta["head_families"] = sorted(hpf)
    meta["sub_families"] = sorted(spf)

    wup = _best_wup(hs, ss)
    meta["wup"] = wup

    # --------------------------------------------------------------
    # Original noun explicitly appears in the question.
    #
    # Coordinate object replacements normally change the requested slot:
    # toilet->sink, leg->arm, charger->cord.
    #
    # Person-to-person is retained as the generic false-presupposition
    # correction needed for cases such as woman->man.
    # Dominant-family filtering ensures woman->chair cannot enter here.
    # --------------------------------------------------------------
    if q_mentions:
        if hfam == "person" and sfam == "person":
            return True, "semantic_person_correction", meta

        return False, "semantic_question_role_change", meta

    shared = hpf & spf

    if shared and wup >= 0.70:
        return True, "semantic_shared_family", meta

    if wup >= 0.86:
        return True, "semantic_close_wordnet", meta

    return False, "semantic_not_close", meta
