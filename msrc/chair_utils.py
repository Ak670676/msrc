# -*- coding: utf-8 -*-
"""
CHAIR utilities: MSCOCO synonym mapping, caption -> mentioned categories,
GT loading from instances_val2014.json, coverage and SWC metrics.

The synonym list follows the standard CHAIR implementation (Rohrbach et al.,
EMNLP 2018) as replicated across POPE/OPERA/VCD codebases. For final paper
numbers, verify against the official synonyms.txt from the CHAIR repo.
"""
import json, os
from collections import defaultdict

from msrc.lattice import singularize, is_ancestor, wn_depth

# category -> synonyms (all lowercase, singular)
SYNONYMS = {
    "person": ["person", "girl", "boy", "man", "woman", "kid", "child",
               "chef", "baker", "people", "adult", "rider", "child", "baby",
               "worker", "passenger", "sister", "biker", "policeman", "cop",
               "officer", "lady", "cowboy", "bride", "groom", "male",
               "female", "guy", "traveler", "mother", "father", "gentleman",
               "pitcher", "player", "skier", "snowboarder", "skater",
               "skateboarder", "foreigner", "caller", "offender", "coworker",
               "trespasser", "patient", "politician", "soldier",
               "grandchild", "serviceman", "walker", "drinker", "doctor",
               "lawyer", "judge", "teacher", "student", "kid", "hiker",
               "climber", "cyclist", "surfer", "pedestrian", "dancer",
               "friend", "couple", "boyfriend", "girlfriend", "cook",
               "waiter", "waitress", "fan", "crowd", "human"],
    "bicycle": ["bicycle", "bike", "unicycle", "minibike", "trike"],
    "car": ["car", "automobile", "van", "minivan", "sedan", "suv", "hatchback",
            "cab", "jeep", "coupe", "taxicab", "limo", "taxi"],
    "motorcycle": ["motorcycle", "scooter", "motorbike", "moped", "vespa"],
    "airplane": ["airplane", "jetliner", "plane", "airbus", "jet", "biplane",
                 "seaplane", "aircraft"],
    "bus": ["bus", "minibus", "trolley"],
    "train": ["train", "locomotive", "tramway", "caboose"],
    "truck": ["truck", "pickup", "lorry", "hauler", "firetruck"],
    "boat": ["boat", "ship", "liner", "sailboat", "motorboat", "dinghy",
             "powerboat", "speedboat", "canoe", "kayak", "yacht", "catamaran",
             "gondola", "ferry", "houseboat", "vessel"],
    "traffic light": ["traffic light", "street light", "traffic signal",
                      "stop light", "streetlight", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench", "pew"],
    "bird": ["bird", "ostrich", "owl", "seagull", "goose", "duck", "parakeet",
             "falcon", "robin", "pelican", "waterfowl", "heron", "hummingbird",
             "mallard", "finch", "pigeon", "sparrow", "seabird", "osprey",
             "blackbird", "fowl", "shorebird", "woodpecker", "egret",
             "chickadee", "quail", "bluebird", "kingfisher", "buzzard",
             "willet", "gull", "swan", "bluejay", "flamingo", "cormorant",
             "parrot", "loon", "gosling", "waterbird", "pheasant", "rooster",
             "sandpiper", "crow", "raven", "turkey", "oriole", "cowbird",
             "warbler", "magpie", "peacock", "cockatiel", "lorikeet",
             "puffin", "vulture", "condor", "macaw", "peafowl", "eagle",
             "bluebird", "hawk", "chicken", "dove"],
    "cat": ["cat", "kitten", "feline", "tabby"],
    "dog": ["dog", "puppy", "beagle", "pup", "chihuahua", "schnauzer",
            "dachshund", "rottweiler", "canine", "pitbull", "collie",
            "pug", "terrier", "poodle", "labrador", "doggie", "doberman",
            "mutt", "doggy", "spaniel", "bulldog", "sheepdog", "weimaraner",
            "corgi", "cocker", "greyhound", "retriever", "brindle", "hound",
            "whippet", "husky"],
    "horse": ["horse", "colt", "pony", "racehorse", "stallion", "equine",
              "mare", "foal", "palomino", "mustang", "clydesdale", "bronc",
              "bronco"],
    "sheep": ["sheep", "lamb", "ram", "goat", "ewe"],
    "cow": ["cow", "cattle", "oxen", "ox", "calf", "holstein", "heifer",
            "buffalo", "bull", "zebu", "bison"],
    "elephant": ["elephant"],
    "bear": ["bear", "panda"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "wallet", "purse", "briefcase"],
    "tie": ["tie", "bow", "bow tie"],
    "suitcase": ["suitcase", "suit case", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat"],
    "baseball glove": ["baseball glove"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "longboard", "skimboard", "shortboard",
                  "wakeboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife", "pocketknife", "knive"],
    "spoon": ["spoon"],
    "bowl": ["bowl", "container"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "sub", "cheeseburger", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut", "bagel"],
    "cake": ["cake", "cheesecake", "cupcake", "shortcake", "coffeecake",
             "pancake"],
    "chair": ["chair", "recliner", "stool", "armchair"],
    "couch": ["couch", "sofa", "loveseat"],
    "potted plant": ["potted plant", "houseplant", "plant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"],
    "toilet": ["toilet", "urinal", "commode", "lavatory", "potty"],
    "tv": ["tv", "television", "televisions", "monitor"],
    "laptop": ["laptop", "computer", "notebook", "netbook", "macbook"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "mobile phone", "phone", "cellphone",
                   "telephone", "phones", "smartphone", "iphone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stovetop", "stove", "furnace"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddybear", "teddy"],
    "hair drier": ["hair drier", "hairdryer", "hair dryer"],
    "toothbrush": ["toothbrush"],
}

WORD2CAT = {}
DOUBLE_WORDS = {}
for cat, syns in SYNONYMS.items():
    for s in syns:
        if " " in s:
            DOUBLE_WORDS[s] = cat
        else:
            WORD2CAT.setdefault(s, cat)


def caption_to_categories(caption):
    """Return the list of COCO categories mentioned (with word evidence)."""
    text = caption.lower().replace(",", " , ").replace(".", " . ")
    words = [singularize(w) for w in text.split()]
    found, i = [], 0
    while i < len(words):
        bigram = " ".join(words[i:i + 2])
        if bigram in DOUBLE_WORDS:
            found.append((DOUBLE_WORDS[bigram], bigram))
            i += 2
            continue
        if words[i] in WORD2CAT:
            found.append((WORD2CAT[words[i]], words[i]))
        i += 1
    return found


def load_gt(instances_path, captions_path=None):
    """image_id -> set of GT category names.

    Standard CHAIR (Rohrbach et al. 2018) derives ground truth from BOTH the
    instance segmentations AND the five human reference captions. Using
    segments alone marks objects humans described but COCO never segmented as
    hallucinations. Defaults to the sibling captions_<split>.json.
    """
    print(f"[chair] loading {instances_path} ...")
    data = json.load(open(instances_path))
    cid2name = {c["id"]: c["name"] for c in data["categories"]}
    gt = defaultdict(set)
    for ann in data["annotations"]:
        gt[ann["image_id"]].add(cid2name[ann["category_id"]])
    if captions_path is None:
        _c = instances_path.replace("instances_", "captions_")
        captions_path = _c if os.path.exists(_c) else None
    if captions_path:
        print(f"[chair] + reference captions {captions_path} ...")
        _n = 0
        for ann in json.load(open(captions_path))["annotations"]:
            for _cat, _w in caption_to_categories(ann["caption"]):
                if _cat not in gt[ann["image_id"]]:
                    gt[ann["image_id"]].add(_cat)
                    _n += 1
        print(f"[chair] caption-derived GT objects added: {_n}")
    else:
        print("[chair] WARNING: captions_*.json not found next to "
              "instances_*.json -- segments-only GT overstates "
              "hallucination vs published CHAIR numbers")
    return gt


def label_word(word, gt_cats):
    """Hallucination label for a mentioned word, hypernym-aware.
    0 = grounded, 1 = hallucinated, None = unlabelable (not a COCO word and
    not an ancestor of any GT category)."""
    w = singularize(word.lower())
    if w in WORD2CAT:
        return 0 if WORD2CAT[w] in gt_cats else 1
    # emitted hypernym: true iff it is an ancestor of some GT category head
    for g in gt_cats:
        head = g.split()[-1]
        if is_ancestor(w, head):
            return 0
    return None


def chair_metrics(captions, gt):
    """captions: list of {'image_id', 'caption'}; returns metric dict."""
    n_cap_hall, n_ment, n_hall_ment = 0, 0, 0
    cov, swc = [], []
    for item in captions:
        img, cap = item["image_id"], item["caption"]
        gt_cats = gt.get(img, set())
        mentioned = caption_to_categories(cap)
        hall = [1 for cat, _ in mentioned if cat not in gt_cats]
        n_ment += len(mentioned)
        n_hall_ment += len(hall)
        n_cap_hall += 1 if hall else 0
        if gt_cats:
            mset = {cat for cat, _ in mentioned}
            cov.append(len(mset & gt_cats) / len(gt_cats))
            # SWC: credit exact/hypernym mentions by relative WordNet depth
            words = {singularize(w) for w in cap.lower().split()}
            per_g = []
            for g in gt_cats:
                head = g.split()[-1]
                best = 0.0
                if g in mset:
                    best = 1.0
                else:
                    dg = max(wn_depth(head), 1)
                    for w in words:
                        if is_ancestor(w, head) and w != head:
                            best = max(best, min(wn_depth(w) / dg, 1.0))
                per_g.append(best)
            swc.append(sum(per_g) / len(per_g))
    n = max(len(captions), 1)
    return {
        "CHAIRs": 100.0 * n_cap_hall / n,
        "CHAIRi": 100.0 * n_hall_ment / max(n_ment, 1),
        "mentions": n_ment,
        "coverage": 100.0 * sum(cov) / max(len(cov), 1),
        "SWC": 100.0 * sum(swc) / max(len(swc), 1),
        "n_captions": len(captions),
    }
