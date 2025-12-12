# ===================================================================== 
# Smart Recipe Finder (PRO) — Distinct Cards + Hard Ingredient Coverage
# =====================================================================

from __future__ import annotations
import io, os, re, json, hashlib, random, time, unicodedata, uuid
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import sklearn

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import requests
import re
import unicodedata

def _normalize_veg_dairy_terms(rec: dict) -> dict:
    """
    Convert dairy/gelatin/ice-cream items to plant-based equivalents.
    Safe for vegetarian mode only.
    """
    if not rec:
        return rec

    # Patterns to replace
    replacements = [
        (r"\bmilk\b",       "soy milk"),
        (r"\byogurt\b",     "soy yogurt"),
        (r"\bbutter\b",     "soy butter"),
        (r"\bcheese\b",     "soy cheese"),
        (r"\bcream\b",      "soy cream"),
        (r"\bice\s*cream\b","soy ice cream"),
        (r"\bgelatin(e)?\b","agar agar"),
    ]

    def sub_all(text: str) -> str:
        t = text.lower()
        for pat, rep in replacements:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        return t

    new = dict(rec)

    # ingredients
    new_ings = []
    for ing in rec.get("ingredients", []):
        new_ings.append(sub_all(str(ing)))
    new["ingredients"] = new_ings

    # steps
    steps = rec.get("steps", [])
    if isinstance(steps, list):
        new["steps"] = [sub_all(str(s)) for s in steps]
    else:
        new["steps"] = sub_all(str(steps))

    # title
    new["title"] = sub_all(rec.get("title",""))

    return new


# --------------------------- Optional stacks ---------------------------
PDF_OK = True; TRANS_OK = True; TTS_OK = True; KALEIDO_OK = True
def _probe_optional():
    global PDF_OK, TRANS_OK, TTS_OK, KALEIDO_OK
    try: import reportlab                    # noqa: F401
    except Exception: PDF_OK = False
    try: from deep_translator import GoogleTranslator  # noqa: F401
    except Exception: TRANS_OK = False
    try: import gtts, pydub                 # noqa: F401
    except Exception: TTS_OK = False
    try: _ = pio.kaleido.scope
    except Exception: KALEIDO_OK = False
_probe_optional()
PDF_OK = True; TRANS_OK = True; TTS_OK = True; KALEIDO_OK = True
def _probe_optional():
    global PDF_OK, TRANS_OK, TTS_OK, KALEIDO_OK
    try: import reportlab                    # noqa: F401
    except Exception: PDF_OK = False
    try: from deep_translator import GoogleTranslator  # noqa: F401
    except Exception: TRANS_OK = False
    try: import gtts, pydub                 # noqa: F401
    except Exception: TTS_OK = False
    try: _ = pio.kaleido.scope
    except Exception: KALEIDO_OK = False

_probe_optional()

# ---- Localized labels for PDFs (en/fr/de) ----
_PDF_I18N = {
    "en": {
        "ingredients": "Ingredients:",
        "instructions": "Instructions:",
        "nutrition": "Nutrition (approx.):",
        "kcal": "Calories (kcal)",
        "protein": "Protein (g)",
        "fat": "Fat (g)",
        "carbs": "Carbs (g)",
        "podcast_title_suffix": "— Podcast Transcript",
    },
    "fr": {
        "ingredients": "Ingrédients :",
        "instructions": "Instructions :",
        "nutrition": "Valeurs nutritionnelles (approx.) :",
        "kcal": "Calories (kcal)",
        "protein": "Protéines (g)",
        "fat": "Lipides (g)",
        "carbs": "Glucides (g)",
        "podcast_title_suffix": "— Transcription du podcast",
    },
    "de": {
        "ingredients": "Zutaten:",
        "instructions": "Anleitung:",
        "nutrition": "Nährwerte (ca.):",
        "kcal": "Kalorien (kcal)",
        "protein": "Eiweiß (g)",
        "fat": "Fett (g)",
        "carbs": "Kohlenhydrate (g)",
        "podcast_title_suffix": "— Podcast-Transkript",
    },
}

def _lbl(lang: str, key: str) -> str:
    d = _PDF_I18N.get(lang or "en", _PDF_I18N["en"])
    return d.get(key, _PDF_I18N["en"].get(key, key))


# --------------------------- Secrets helper ---------------------------
def _get_env_or_secret(name: str) -> str:
    val = os.environ.get(name, "")
    if val: return val.strip()
    try: return str(st.secrets[name]).strip()
    except Exception: return ""

# ------------------------------ Paths ---------------------------------
APP_DIR     = Path(__file__).resolve().parent
ASSETS_DIR  = APP_DIR / "assets"
LOGO_PATH   = ASSETS_DIR / "logo_srf.png"
ICON_VEG    = ASSETS_DIR / "veg.png"
ICON_NONV   = ASSETS_DIR / "nonveg.png"

MODEL_PATHS = [
    APP_DIR / "cuisine_pipeline.joblib",
    APP_DIR / "models" / "cuisine_pipeline.joblib",
    Path("/mnt/data") / "cuisine_pipeline.joblib",
]
LABEL_PATHS = [
    APP_DIR / "labels.json",
    APP_DIR / "models" / "labels.json",
    Path("/mnt/data") / "labels.json",
]

# placeholders used only if image APIs fail; lock seeded by hash(title+salt)
IMG_VEG_FALLBACK = "vegetarian,kebab,grill,dish"
IMG_NON_FALLBACK = "bbq,kebab,grill,meal"

TITLE = "Smart Recipe Finder (PRO)"
TOPK  = 3
np.random.seed(42)

LANGUAGE_CHOICES = {"English":"en","French":"fr","German":"de"}

st.set_page_config(page_title=TITLE, page_icon="🍽️", layout="wide")
st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"]{background:#dc2626;border-color:#dc2626}
    [data-testid="stExpander"] details summary{font-weight:600}
    .card {border-radius:10px;background:#0b0f14;padding:10px;margin-bottom:8px}
    .pill {background:#065f46;color:white;padding:6px 10px;border-radius:8px;display:inline-block}
    .caption {font-size:0.78rem;opacity:0.65}
    .howto li {margin-bottom:4px;}
    </style>
    """,
    unsafe_allow_html=True
)

if LOGO_PATH.exists():
    _, c2, _ = st.columns([1,2,1])
    with c2: st.image(str(LOGO_PATH), width=240)

# --------------------------- Session defaults --------------------------
for k, v in {
    "meal_plan": [], "pred_ready": False, "ings": [], "df_pred": pd.DataFrame(),
    "cuisines": [], "meta": {}, "selected": None, "last_sig": "",
    "recs_top3": [], "diet": "Non-vegetarian", "podcast_context": {},
    "veg_dropped": set(), "title_to_ings": {}, "search_salt":""
}.items(): st.session_state.setdefault(k, v)

# --------------------------- Nutrition (toy) ---------------------------
# --------------------------- Nutrition (toy) ---------------------------
# Canonical nutrition + rich alias coverage (all per 100g)
NUTR_TABLE = {
    # --- Chicken & Poultry ---
    "chicken":               {"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "chicken breast":        {"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "boneless chicken":      {"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "chicken fillet":        {"kcal":165,"protein":31.0,"fat":3.6,"carbs":0.0},
    "chicken thighs":        {"kcal":209,"protein":26.0,"fat":10.9,"carbs":0.0},
    "chicken thigh":         {"kcal":209,"protein":26.0,"fat":10.9,"carbs":0.0},
    "chicken drumstick":     {"kcal":161,"protein":18.0,"fat":9.2,"carbs":0.0},
    "drumstick":             {"kcal":161,"protein":18.0,"fat":9.2,"carbs":0.0},
    "chicken wing":          {"kcal":203,"protein":30.5,"fat":8.1,"carbs":0.0},
    "wings":                 {"kcal":203,"protein":30.5,"fat":8.1,"carbs":0.0},
    "ground chicken":        {"kcal":189,"protein":23.0,"fat":10.0,"carbs":0.0},

    "turkey":                {"kcal":135,"protein":29.0,"fat":1.0,"carbs":0.0},
    "turkey breast":         {"kcal":135,"protein":29.0,"fat":1.0,"carbs":0.0},
    "ground turkey":         {"kcal":209,"protein":27.0,"fat":10.0,"carbs":0.0},

    "duck":                  {"kcal":337,"protein":19.0,"fat":28.0,"carbs":0.0},
    "duck breast":           {"kcal":337,"protein":19.0,"fat":28.0,"carbs":0.0},
    "goose":                 {"kcal":305,"protein":25.0,"fat":22.0,"carbs":0.0},

    # --- Beef ---
    "beef":                  {"kcal":217,"protein":26.1,"fat":11.8,"carbs":0.0},
    "ground beef":           {"kcal":254,"protein":26.0,"fat":15.0,"carbs":0.0},
    "minced beef":           {"kcal":254,"protein":26.0,"fat":15.0,"carbs":0.0},
    "beef mince":            {"kcal":254,"protein":26.0,"fat":15.0,"carbs":0.0},
    "sirloin":               {"kcal":206,"protein":28.0,"fat":10.0,"carbs":0.0},
    "ribeye":                {"kcal":291,"protein":24.0,"fat":21.0,"carbs":0.0},
    "steak":                 {"kcal":217,"protein":26.1,"fat":11.8,"carbs":0.0},

    # --- Lamb / Goat / Game ---
    "lamb":                  {"kcal":294,"protein":25.0,"fat":21.0,"carbs":0.0},
    "ground lamb":           {"kcal":294,"protein":25.0,"fat":21.0,"carbs":0.0},
    "minced lamb":           {"kcal":294,"protein":25.0,"fat":21.0,"carbs":0.0},
    "goat":                  {"kcal":143,"protein":27.0,"fat":3.0,"carbs":0.0},
    "venison":               {"kcal":158,"protein":30.0,"fat":3.2,"carbs":0.0},

    # --- Pork ---
    "pork":                  {"kcal":242,"protein":27.0,"fat":14.0,"carbs":0.0},
    "pork loin":             {"kcal":242,"protein":27.0,"fat":14.0,"carbs":0.0},
    "pork chop":             {"kcal":242,"protein":27.0,"fat":14.0,"carbs":0.0},
    "pork tenderloin":       {"kcal":143,"protein":26.0,"fat":3.5,"carbs":0.0},
    "pork belly":            {"kcal":518,"protein":9.3,"fat":53.0,"carbs":0.0},
    "pork sausage":          {"kcal":301,"protein":12.0,"fat":27.0,"carbs":1.0},
    "bacon":                 {"kcal":541,"protein":37.0,"fat":42.0,"carbs":1.4},

    # --- Processed ---
    "ham":                   {"kcal":145,"protein":20.9,"fat":5.5,"carbs":1.5},
    "salami":                {"kcal":336,"protein":22.0,"fat":26.0,"carbs":1.5},

    # --- Seafood & Aliases ---
    "whitefish":             {"kcal":96,"protein":20.0,"fat":1.5,"carbs":0.0},
    "fish":                  {"kcal":96,"protein":20.0,"fat":1.5,"carbs":0.0},
    "seafood":               {"kcal":96,"protein":20.0,"fat":1.5,"carbs":0.0},

    "salmon":                {"kcal":208,"protein":20.4,"fat":13.0,"carbs":0.0},
    "salmon fillet":         {"kcal":208,"protein":20.4,"fat":13.0,"carbs":0.0},

    "cod":                   {"kcal":82,"protein":18.0,"fat":0.7,"carbs":0.0},
    "haddock":               {"kcal":90,"protein":19.9,"fat":0.6,"carbs":0.0},
    "trout":                 {"kcal":190,"protein":26.0,"fat":8.0,"carbs":0.0},
    "tilapia":               {"kcal":129,"protein":26.1,"fat":2.7,"carbs":0.0},
    "tuna":                  {"kcal":132,"protein":29.0,"fat":1.0,"carbs":0.0},
    "canned tuna":           {"kcal":132,"protein":29.0,"fat":1.0,"carbs":0.0},
    "mackerel":              {"kcal":205,"protein":18.6,"fat":13.9,"carbs":0.0},
    "sardine":               {"kcal":208,"protein":24.6,"fat":11.5,"carbs":0.0},
    "anchovy":               {"kcal":210,"protein":29.0,"fat":10.0,"carbs":0.0},

    "shrimp":                {"kcal":99,"protein":24.0,"fat":0.3,"carbs":0.2},
    "prawn":                 {"kcal":99,"protein":24.0,"fat":0.3,"carbs":0.2},
    "scallop":               {"kcal":88,"protein":16.8,"fat":0.8,"carbs":3.2},
    "squid":                 {"kcal":92,"protein":15.6,"fat":1.4,"carbs":3.1},
    "octopus":               {"kcal":82,"protein":14.9,"fat":1.0,"carbs":2.2},
    "clam":                  {"kcal":86,"protein":14.7,"fat":0.9,"carbs":3.6},
    "mussel":                {"kcal":172,"protein":24.0,"fat":4.5,"carbs":7.4},
    "crab":                  {"kcal":97,"protein":21.0,"fat":1.5,"carbs":0.0},
    "lobster":               {"kcal":89,"protein":19.0,"fat":1.0,"carbs":0.0},

    # --- Soy / Legumes ---
    "tofu":                  {"kcal":76,"protein":8.0,"fat":4.8,"carbs":1.9},
    "lentil":                {"kcal":116,"protein":9.0,"fat":0.4,"carbs":20.1},
    "bean":                  {"kcal":127,"protein":8.7,"fat":0.5,"carbs":22.8},
    "chickpea":              {"kcal":164,"protein":8.9,"fat":2.6,"carbs":27.4},

    # --- Mushrooms (full alias set) ---
    "mushroom":              {"kcal":22,"protein":3.1,"fat":0.3,"carbs":3.3},
    "button mushroom":       {"kcal":22,"protein":3.1,"fat":0.3,"carbs":3.3},
    "cremini":               {"kcal":22,"protein":3.1,"fat":0.3,"carbs":3.3},
    "portobello":            {"kcal":22,"protein":3.1,"fat":0.3,"carbs":3.3},
    "shiitake":              {"kcal":34,"protein":2.2,"fat":0.5,"carbs":6.8},
    "oyster mushroom":       {"kcal":33,"protein":3.3,"fat":0.4,"carbs":6.1},
    "king oyster":           {"kcal":35,"protein":3.5,"fat":0.3,"carbs":6.2},
    "enoki":                 {"kcal":37,"protein":2.7,"fat":0.3,"carbs":7.8},
    "maitake":               {"kcal":31,"protein":1.9,"fat":0.2,"carbs":6.9},
    "chanterelle":           {"kcal":38,"protein":1.5,"fat":0.5,"carbs":6.9},
    "porcini":               {"kcal":26,"protein":3.1,"fat":0.1,"carbs":4.6},
    "morel":                 {"kcal":31,"protein":3.1,"fat":0.6,"carbs":5.1},

    # --- Oils & Staples ---
    "olive oil":             {"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "sesame oil":            {"kcal":884,"protein":0.0,"fat":100.0,"carbs":0.0},
    "butter":                {"kcal":717,"protein":0.9,"fat":81.1,"carbs":0.1},
    "soy sauce":             {"kcal":53,"protein":8.0,"fat":0.6,"carbs":5.6},
    "rice":                  {"kcal":130,"protein":2.4,"fat":0.3,"carbs":28.0},
    "pasta":                 {"kcal":131,"protein":5.0,"fat":1.1,"carbs":25.0},

    # --- Vegetables & Herbs ---
    "garlic":                {"kcal":149,"protein":6.4,"fat":0.5,"carbs":33.1},
    "ginger":                {"kcal":80,"protein":1.8,"fat":0.8,"carbs":17.8},
    "onion":                 {"kcal":40,"protein":1.1,"fat":0.1,"carbs":9.3},
    "tomato":                {"kcal":18,"protein":0.9,"fat":0.2,"carbs":3.9},
    "cherry tomatoes":       {"kcal":18,"protein":0.9,"fat":0.2,"carbs":3.9},
    "basil":                 {"kcal":23,"protein":3.2,"fat":0.6,"carbs":2.7},
    "spinach":               {"kcal":23,"protein":2.9,"fat":0.4,"carbs":3.6},
    "broccoli":              {"kcal":34,"protein":2.8,"fat":0.4,"carbs":6.6}
}


# ------------------------------ Model ---------------------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists(): return p
    return None

@st.cache_resource(show_spinner="Loading cuisine model…")
def load_pipeline():
    mp = _first_existing(MODEL_PATHS)
    lp = _first_existing(LABEL_PATHS)
    if mp is None: st.error("Missing model file: cuisine_pipeline.joblib"); st.stop()
    if lp is None: st.error("Missing labels file: labels.json"); st.stop()
    pipe = joblib.load(mp)
    raw  = json.loads(lp.read_text(encoding="utf-8"))
    inv  = {i:n for i,n in enumerate(raw)} if isinstance(raw, list) else {int(k):v for k,v in raw.items()}
    return pipe, inv

pipe, INV = load_pipeline()

# ------------------------------ Helpers -------------------------------
# ============================================================
# ============================================================
# Strict Non-Vegetarian tokens (used by _near_nonveg / veg gate)
# Eggs & gelatin included (strict-veg per your UI description)
# ============================================================
NONVEG = {
    # generic
    "meat","seafood",

    # poultry & livestock
    "chicken","beef","pork","lamb","mutton","turkey","duck","veal","goat",

    # game meats
    "venison","boar","rabbit","kangaroo",

    # processed & cured meats / fats
    "bacon","ham","prosciutto","mortadella","pastrami","bresaola","capicola",
    "pepperoni","salami","chorizo","sausage","frankfurter","hotdog","lard",

    # organ meats
    "liver","kidney","tripe","sweetbread","offal","liverwurst",

    # fish (broad coverage)
    "fish","salmon","tuna","cod","haddock","pollock","tilapia","catfish","herring",
    "anchovy","sardine","mackerel","trout","bass","snapper","swordfish","shark","eel",

    # cephalopods & molluscs
    "squid","calamari","octopus","cuttlefish","oyster","clam","mussel","scallop",

    # crustaceans
    "shrimp","prawn","crab","lobster","krill",

    # eggs & gelatin (strict vegetarian excludes these)
    "egg","eggs","yolk","gelatin","gelatine"
}

# ============================================================
# Common spelling mistakes / alias correction (lowercase keys)
# (Used by _canon_token before NONVEG checks)
# ============================================================
NONVEG_ALIASES = {
    # poultry / livestock misspellings
    "chiken":"chicken","chikn":"chicken","chk":"chicken","ckn":"chicken",
    "beaf":"beef","biff":"beef","porc":"pork","porco":"pork","lam":"lamb",
    "mouton":"mutton","turky":"turkey","duk":"duck","goatte":"goat",

    # processed meats
    "becon":"bacon","bacn":"bacon","jamon":"ham","prosciuto":"prosciutto",
    "pepperonni":"pepperoni","peperoni":"pepperoni","salame":"salami",
    "saussage":"sausage","sossage":"sausage","chorizzo":"chorizo",
    "frankfurter":"frankfurter","hot dog":"hotdog","hot-dog":"hotdog",

    # fish & seafood misspellings/aliases
    "fishes":"fish","tuna fish":"tuna","salomon":"salmon","salamon":"salmon",
    "mackarel":"mackerel","anchovi":"anchovy","anchovey":"anchovy",
    "sardin":"sardine","sardines":"sardine","codfish":"cod",
    "basa":"catfish",  # common market name mapping (optional)
    "calamary":"calamari","sqiud":"squid","octopuss":"octopus",
    "prawns":"prawn","shrimpes":"shrimp","shrimps":"shrimp",

    # organ/offal
    "foie":"liver","foiegras":"liver","sweetbreads":"sweetbread",

    # eggs & gelatin
    "eg":"egg","yolks":"yolk","gelatine":"gelatin",

    # plurals to base
    "meats":"meat","beefs":"beef","chickens":"chicken","fishes":"fish",
    "sausages":"sausage","bacons":"bacon","hams":"ham","salamis":"salami",
    "pepperonis":"pepperoni","prawns":"prawn","shrimps":"shrimp",
    "lobsters":"lobster","crabs":"crab","oysters":"oyster","clams":"clam",
    "mussels":"mussel","scallops":"scallop","eggs":"egg"
}


# ============================================================
# Non-veg phrases (sauces, stocks, pastes, broths)
# Dairy & eggs are intentionally NOT here.
# ============================================================
_NONVEG_PHRASES = [
    "fish sauce", "oyster sauce", "shrimp paste", "bonito", "nam pla", "dashi",
    "chicken stock", "beef stock", "fish stock", "bone broth",
]

# --- Vegetarian substitutions ---
_VEG_SUBS = {
    r"\bmilk\b": ["soy milk", "almond milk", "oat milk"],
    r"\byogurt\b": ["soy yogurt", "coconut yogurt", "almond yogurt"],
    r"\bbutter\b": ["soy butter", "olive oil", "vegan butter"],
    r"\bcheese\b": ["soy cheese", "plant-based cheese", "nut parmesan"],
    r"\bcream\b": ["soy cream", "coconut cream", "cashew cream"],
    r"\bfish sauce\b": ["soy sauce + lime", "mushroom soy"],
    r"\bchicken stock\b": ["vegetable stock"],
    r"\bbeef stock\b": ["vegetable stock"],
    r"\bfish stock\b": ["vegetable stock"],
    r"\begg\b|\beggs\b|\byolk\b|\byolks\b": ["flax egg", "chia egg"],
    r"\bchicken\b": ["firm tofu", "seitan"],
    r"\bbeef\b": ["mushroom mince", "textured soy"],
    r"\bpork\b": ["tempeh"],
    r"\blamb\b|\bmutton\b": ["seitan"],
    r"\bfish\b|\bsalmon\b|\btuna\b|\bcod\b": ["tofu (pressed)", "king oyster mushroom"],
    r"\bshrimp\b|\bprawn\b|\bcrab\b": ["king oyster mushroom slices"],
}

_DAIRY_REQ = {"milk", "yogurt", "butter", "cheese", "cream"}

def _rng_choice_deterministic(options: list[str], salt: str) -> str:
    if not options: return ""
    h = int(hashlib.sha1(("subs|" + salt).encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]

def _apply_veg_substitutions_text(text: str, salt: str) -> str:
    out = text
    for pat, opts in _VEG_SUBS.items():
        repl = _rng_choice_deterministic(opts, salt)
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

def _apply_veg_substitutions_list(items: list[str], salt: str) -> list[str]:
    return [_apply_veg_substitutions_text(str(x), salt) for x in items]
def _apply_veg_substitutions_recipe(rec: dict, salt: str) -> dict:
    """Apply vegetarian-friendly substitutions to a whole recipe object."""
    r = dict(rec)
    r["ingredients"] = _apply_veg_substitutions_list(r.get("ingredients", []), salt)
    steps = r.get("steps", [])
    if isinstance(steps, list):
        r["steps"] = [_apply_veg_substitutions_text(str(s), salt) for s in steps]
    else:
        r["steps"] = _apply_veg_substitutions_text(str(steps), salt)
    r["title"] = _apply_veg_substitutions_text(r.get("title", ""), salt)
    return r


def _lev1(a: str, b: str) -> int:
    if a == b: return 0
    if abs(len(a)-len(b)) > 1: return 2
    if len(a) == len(b):
        mism = sum(1 for x, y in zip(a, b) if x != y)
        return 1 if mism == 1 else 2
    if len(a) < len(b): a, b = b, a
    for i in range(len(a)):
        if a[:i] + a[i+1:] == b: return 1
    return 2

def _canon_token(t: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", t.lower())
    if s in NONVEG_ALIASES: s = NONVEG_ALIASES[s]
    if s.endswith("s") and len(s) > 3: s = s[:-1]
    return s

def _near_nonveg(tok: str) -> bool:
    c = _canon_token(tok)
    if c in NONVEG: return True
    for nv in NONVEG:
        if _lev1(c, nv) <= 1: return True
    return False

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()

def _tokset(xs: list[str]) -> set[str]:
    out = set()
    for x in xs:
        for t in re.sub(r"[^a-z0-9\s]"," ",str(x).lower()).split():
            if len(t) >= 2:
                out.add(_canon_token(t))
    return out

def _best3(xs: List[str]) -> str:
    base = [i.strip() for i in xs if i.strip()][:3]
    return ", ".join(base) if base else "seasonal ingredients"

def predict_topk(pipe, inv_labels, ings: List[str], k: int = TOPK) -> Tuple[List[str], np.ndarray]:
    txt = " ".join(ings)
    proba = pipe.predict_proba([txt])[0]
    order = np.argsort(proba)[::-1]
    names, values, seen = [], [], set()
    for idx in order:
        c = inv_labels[idx]
        if c not in seen:
            names.append(c); values.append(float(proba[idx])); seen.add(c)
        if len(names) == k: break
    return names, np.array(values, dtype=float)

def macro_totals(items: List[str]) -> Dict[str, float]:
    tot = {"kcal":0.0,"protein":0.0,"fat":0.0,"carbs":0.0}
    for it in items:
        s = it.lower()
        for key, nt in NUTR_TABLE.items():
            if key in s:
                tot["kcal"] += nt["kcal"]; tot["protein"] += nt["protein"]
                tot["fat"]  += nt["fat"];   tot["carbs"]   += nt["carbs"]
                break
    return tot

def _canon_ings(text: str) -> tuple[list[str], str]:
    text = unicodedata.normalize("NFKC", str(text))
    tokens = re.split(r"[,\n;؛]+", text)
    ings = [t.strip() for t in tokens if t and t.strip()]
    sig = ",".join(sorted([s.lower() for s in ings]))
    return ings, sig

def _veg_milk_swap(ings: list[str]) -> tuple[list[str], set[str]]:
    """
    Vegetarian mode: drop non-plant dairy terms and inject soy equivalents once.
    Handles milk, yogurt/yoghurt, butter, cheese, cream.
    Returns (new_ings, swapped_originals_set).
    """
    out: list[str] = []
    swapped: set[str] = set()

    DAIRY = ("milk", "yogurt", "yoghurt", "butter", "cheese", "cream")
    PLANT = ("soy", "soya", "almond", "oat", "coconut", "rice", "plant", "vegan")

    for it in ings:
        low = it.strip().lower()
        is_dairy = any(w in low for w in DAIRY)
        is_plant = any(p in low for p in PLANT)
        if is_dairy and not is_plant:
            swapped.add(it.strip())   # remove animal-dairy item
        else:
            out.append(it)

    # Inject soy- equivalents based on what user originally mentioned
    blob = " ".join(ings).lower()

    def _ensure(term: str):
        if not any(term in x.lower() for x in out):
            out.append(term)

    if "milk"   in blob: _ensure("soy milk")
    if "yogurt" in blob or "yoghurt" in blob: _ensure("soy yogurt")
    if "butter" in blob: _ensure("soy butter")
    if "cheese" in blob: _ensure("soy cheese")
    if "cream"  in blob: _ensure("soy cream")

    return out, swapped


# ===== STRICT-VEG helpers & constants (drop-in) =======================

_DAIRY_WORDS    = ("milk","yogurt","yoghurt","butter","cheese","cream")
_STOCK_PHRASES  = ("chicken stock","beef stock","fish stock","bone broth")
_PLANT_MARKERS  = ("soy","soya","almond","oat","coconut","rice","plant","vegan")  # used for detection

def _veg_dairy_swap(items: list[str]) -> tuple[list[str], set[str]]:
    """
    Remove animal-dairy and animal-stocks if they're not already plant-based,
    and inject canonical soy-... replacements (once). Returns (new_items, swapped_set).
    """
    out, swapped = [], set()
    for it in items:
        low = it.lower()

        # Is it a dairy term without a plant marker?
        is_dairy    = any(w in low for w in _DAIRY_WORDS)
        is_plantish = any(p in low for p in _PLANT_MARKERS)

        # Is it an explicit non-veg stock/broth?
        is_stock    = any(ph in low for ph in _STOCK_PHRASES)

        if (is_dairy and not is_plantish) or is_stock:
            swapped.add(it)
        else:
            out.append(it)

    # Inject canonical soy-… once, based on what appeared in the original blob
    blob = " ".join(items).lower()
    inject = []
    if "milk"   in blob and "soy milk"   not in (x.lower() for x in out): inject.append("soy milk")
    if ("yogurt" in blob or "yoghurt" in blob) and "soy yogurt" not in (x.lower() for x in out): inject.append("soy yogurt")
    if "butter" in blob and "soy butter" not in (x.lower() for x in out): inject.append("soy butter")
    if "cheese" in blob and "soy cheese" not in (x.lower() for x in out): inject.append("soy cheese")
    if "cream"  in blob and "soy cream"  not in (x.lower() for x in out): inject.append("soy cream")

    # Always replace animal stocks with vegetable stock
    if any(ph in blob for ph in _STOCK_PHRASES) and "vegetable stock" not in (x.lower() for x in out):
        inject.append("vegetable stock")

    for a in inject:
        out.append(a)

    return out, swapped


def _veg_milk_swap_in_recipe(rec: dict) -> tuple[dict, set[str]]:
    """
    Vegetarian mode post-filter on a single recipe dict:
    - Remove animal dairy (milk, yogurt, butter, cheese, cream) unless already plant-based.
    - Remove animal stocks (chicken/beef/fish stock, bone broth).
    - Inject canonical replacements: soy milk/yogurt/butter/cheese/cream, vegetable stock.
    """
    if not rec:
        return rec, set()
    ings = list(rec.get("ingredients", []))
    new_ings, swapped = _veg_dairy_swap(ings)
    if swapped:
        rec = dict(rec)
        rec["ingredients"] = new_ings
    return rec, swapped


def _is_vegetarian(ingredients: list[str]) -> bool:
    """
    Reject if any token is (near) non-veg OR if any explicit non-veg phrase appears.
    (Relies on global NONVEG, _NONVEG_PHRASES, _tokset, _near_nonveg.)
    """
    tokens = _tokset(ingredients)
    if any(_near_nonveg(t) for t in tokens):
        return False
    low = " ".join(ingredients).lower()
    if any(p in low for p in _NONVEG_PHRASES):
        return False
    # Also treat explicit animal stocks as non-veg
    if any(ph in low for ph in _STOCK_PHRASES):
        return False
    return True


def _required_tokens_for_mode(
    user_ings: list[str],
    vegetarian: bool
) -> tuple[set[str], set[str], set[str]]:
    """
    Build the set of required tokens from user input.
    In vegetarian mode:
      - Drop animal proteins, eggs, gelatin, and explicit animal stocks.
      - Map dairy to 'soy + <word>' (milk→soy milk, cheese→soy cheese, cream→soy cream, ...)
    Also returns:
      - dropped: what got removed (for UI)
      - requested_proteins: exact protein tokens user asked for (fish, chicken, beef, shrimp, ...)
    """
    req, dropped = set(), set()
    requested_proteins: set[str] = set()

    for raw in user_ings:
        low_raw = str(raw).lower().strip()

        # tokenize text
        for t in re.sub(r"[^a-z0-9\s]", " ", low_raw).split():
            if len(t) < 2:
                continue

            c = _canon_token(t)

            if vegetarian:
                # --- drop animal stocks ---
                if any(ph in low_raw for ph in _STOCK_PHRASES):
                    dropped.add(raw.strip())
                    req.update({"vegetable", "stock"})
                    continue

                # --- dairy → soy + word ---
                if any(w in low_raw for w in _DAIRY_WORDS):
                    dropped.add(raw.strip())
                    if "milk"   in low_raw: req.update({"soy", "milk"})
                    if "yogurt" in low_raw or "yoghurt" in low_raw: req.update({"soy", "yogurt"})
                    if "butter" in low_raw: req.update({"soy", "butter"})
                    if "cheese" in low_raw: req.update({"soy", "cheese"})
                    if "cream"  in low_raw: req.update({"soy", "cream"})
                    continue

                # skip bare milk token
                if c == "milk":
                    dropped.add(raw.strip())
                    continue

                # --- drop all non-veg tokens ---
                if _near_nonveg(c) or any(p in low_raw for p in _NONVEG_PHRASES):
                    dropped.add(raw.strip())
                    continue

            # default path
            req.add(c)

            # track exact protein requests
            if c in PROTEIN_TOKENS:
                requested_proteins.add(c)

    return req, dropped, requested_proteins



# ---- Images
# ---------------------- Image retrieval (food-only safe) ----------------------
PEXELS_KEY = _get_env_or_secret("PEXELS_API_KEY")

@st.cache_data(show_spinner=False, ttl=3600)
def _pexels_image(query: str, per_page: int = 18, page: int = 1) -> Optional[str]:
    """Try to fetch a plated/food dish from Pexels."""
    if not PEXELS_KEY:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_KEY, "User-Agent": "SRF/1.0"},
            params={
                "query": query,
                "per_page": per_page,
                "page": page,
                "orientation": "landscape",
                "size": "medium",
            },
            timeout=10,
        )
        r.raise_for_status()
        js = r.json() or {}
        photos = js.get("photos") or []
        # Prefer items whose ALT clearly indicates cooked/plated food.
        food_terms = {
            "plated", "plate", "dish", "meal", "kebab", "skewer", "grill", "bbq",
            "tandoori", "curry", "pasta", "salad", "soup", "pilaf", "rice",
            "noodles", "steak", "burger", "taco", "sushi", "biryani", "shawarma"
        }
        for p in photos:
            alt = (p.get("alt") or "").lower()
            if any(k in alt for k in food_terms):
                src = (p.get("src") or {})
                return src.get("medium") or src.get("large") or src.get("original")
        # Fallback to the first result if nothing matched the strict filter.
        if photos:
            src = (photos[0].get("src") or {})
            return src.get("medium") or src.get("large") or src.get("original")
    except Exception:
        return None
    return None

def _valid_food_img(url: Optional[str]) -> bool:
    """Basic sanity checks; reject obvious non-food or malformed URLs."""
    if not url:
        return False
    u = url.lower()
    reject = [
        "monster energy", "can of", "soda", "soft drink", "coffee cup",
        "fruit", "berries", "banana", "orange", "apple", "juice"
    ]
    return u.startswith(("http://", "https://")) and not any(b in u for b in reject)

# Deterministic, image-only FOOD fallbacks (no people/streets/objects).
FALLBACK_VEG_IMAGES = [
    "https://img.icons8.com/color/512/salad.png",
    "https://img.icons8.com/color/512/vegetarian-food.png",
    "https://img.icons8.com/color/512/greek-salad.png",
    "https://img.icons8.com/color/512/vegan-food.png",
]
FALLBACK_NONVEG_IMAGES = [
    "https://img.icons8.com/color/512/steak.png",
    "https://img.icons8.com/color/512/bbq.png",
    "https://img.icons8.com/color/512/chicken.png",
    "https://img.icons8.com/color/512/fish-food.png",
]

def _fallback_food_image(vegetarian: bool, title: str, salt: str, seen: set[str]) -> str:
    """Guaranteed safe food icon; deterministic w.r.t. (title, salt)."""
    pool = FALLBACK_VEG_IMAGES if vegetarian else FALLBACK_NONVEG_IMAGES
    h = int(hashlib.sha1((title + "|" + salt).encode("utf-8")).hexdigest(), 16)
    idx = h % len(pool)
    # Avoid duplicates inside a single search session.
    for j in range(len(pool)):
        url = pool[(idx + j) % len(pool)]
        if url not in seen:
            seen.add(url)
            return url
    return pool[idx]

def _resolve_image(img: Optional[str], vegetarian: bool, title: str, salt: str, seen: set[str]) -> str:
    """
    Rule:
      1) If the recipe already has a valid food image and it's not reused -> keep it.
      2) Else query Pexels with salted pagination for diversity.
      3) Else use a guaranteed-safe food fallback (no loremflickr).
    """
    # 1) Provided image
    if img and _valid_food_img(img) and img not in seen:
        seen.add(img)
        return img

    # 2) Pexels query (salt controls page selection -> fresh images each search)
    q = (title or "recipe").lower()
    if any(k in q for k in ["bbq", "barbecue", "grill", "grilled", "kebab", "kabob", "skewer", "tandoori"]):
        q += " plated grill kebab barbecue"
    else:
        q += " plated main course curry pasta salad soup pilaf"
        if vegetarian:
            q += " vegetarian"
    page = 1 + (int(hashlib.sha1((q + "|" + salt).encode("utf-8")).hexdigest(), 16) % 3)
    px = _pexels_image(q, per_page=18, page=page)
    if px and px not in seen and _valid_food_img(px):
        seen.add(px)
        return px

    # 3) Guaranteed-safe fallback (icons8 food set)
    return _fallback_food_image(vegetarian, title or q, salt, seen)


# ---------------------- Step generator (distinct per card) -------------
def distinct_steps_from_ingredients(title: str,
                                    ings: list[str],
                                    style: str,
                                    seed: int,
                                    required_tokens: Optional[set[str]]=None) -> list[str]:
    rng = random.Random(seed)
    low_join = " ".join(_norm(x) for x in ings)
    has_tomato  = any(k in low_join for k in ["tomato","passata"])
    has_dairy   = any(k in low_join for k in ["cream","yogurt","cheese","parmesan","paneer","feta","milk"])
    has_meat    = any(_near_nonveg(t) for t in _tokset(ings))
    arom        = [w for w in ings if any(x in _norm(w) for x in ["garlic","onion","ginger","shallot","leek"])]
    fat         = [w for w in ings if any(x in _norm(w) for x in ["olive oil","oil","butter","ghee","sesame oil"])]

    pan = rng.choice(["skillet","Dutch oven","saucepan","braiser"])
    steps: list[str] = []
    steps.append("Prep produce and seasoning.")
    if required_tokens:
        req_line = ", ".join(sorted(required_tokens))
        steps.append(f"Keep required items handy: {req_line}.")

    if style in {"skillet","one-pot pasta","pilaf","curry"}:
        steps.append(f"Preheat {pan} on medium; add {rng.choice(fat) if fat else 'oil'}.")
        if arom:
            steps.append(f"Sweat {', '.join(arom[:2])} with a pinch of salt until fragrant.")

    if style == "one-pot pasta":
        steps += [
            "Add dry pasta; stir to coat.",
            f"Pour in { 'tomato passata' if has_tomato else 'stock/water' } to cover; simmer, stirring.",
            "Finish glossy with starchy liquid; fold through herbs/cheese."
        ]
    elif style == "pilaf":
        steps += [
            "Rinse rice until clear; toast in fat 1–2 min.",
            "Add 1.8× hot stock; cover and simmer gently 12–15 min.",
            "Steam off heat 5 min; fluff."
        ]
    elif style == "curry":
        steps += [
            "Bloom ground spices 30–60 s.",
            f"Add tomatoes and { 'yogurt/cream' if has_dairy else 'water' }; simmer to nappé.",
            "Balance with citrus; finish with herbs."
        ]
    elif style == "tray bake":
        steps += [
            "Preheat oven to 210 °C (410 °F).",
            "Toss components with oil, salt, pepper; spread on tray.",
            "Roast to caramelization, tossing once."
        ]
    elif style == "grill":
        steps += [
            "Preheat grill/pan to high; oil grates.",
            "Thread or arrange components; pat dry and season.",
            "Grill to char marks, basting as needed; finish to doneness."
        ]
    elif style == "stir fry":
        steps += [
            "Heat wok to smoking; add oil.",
            "Stir-fry aromatics; add items in batches for sear.",
            "Deglaze (soy/citrus); thicken lightly if needed."
        ]
    elif style == "soup":
        steps += [
            "Sweat base vegetables; cover with liquid.",
            "Simmer until flavors meld; season.",
            "Finish with fresh herbs."
        ]
    elif style == "salad":
        steps += [
            "Whisk dressing.",
            "Toss chopped components just before serving.",
            "Top with herbs/cheese/seeds."
        ]
    else:  # skillet default
        steps += [
            f"Sear {'protein' if has_meat else 'vegetables'}; deglaze with { 'tomato' if has_tomato else 'stock' }.",
            "Simmer to desired thickness."
        ]

    finishers = ["adjust salt and acidity"]
    if has_dairy:  finishers.append("fold dairy off heat")
    if has_tomato: finishers.append("balance with a pinch of sugar or vinegar")
    steps.append("Finally " + "; ".join(finishers) + ".")
    steps.append("Rest briefly; garnish and serve.")

    # ensure distinct, trimmed, ≤10
    seen, clean = set(), []
    for s in steps:
        s2 = re.sub(r"\s+", " ", s).strip()
        if s2 and s2 not in seen:
            clean.append(s2); seen.add(s2)
        if len(clean) == 10: break
    return clean

# --------------------------- Local Search Engine -----------------------
class LocalRecipeIndex:
    def __init__(self, df: Optional[pd.DataFrame]):
        self.ok=False; self.df=pd.DataFrame()
        if df is None or df.empty: return
        use=df.copy()
        for col in ["title","ingredients","steps","image","diet"]:
            use[col]=use.get(col,pd.Series(dtype=str)).fillna("")
        use["blob"]=(use["title"].astype(str)+" "+
                     use["ingredients"].astype(str)+" "+
                     use["steps"].astype(str)).str.lower()
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer=TfidfVectorizer(min_df=1,max_df=0.9,ngram_range=(1,2))
            self.feats=self.vectorizer.fit_transform(use["blob"].tolist())
            self.df=use.reset_index(drop=True); self.ok=True
        except Exception: self.ok=False

    @staticmethod
    def _ings_query(ingredients: list[str]) -> str:
        toks=[]
        for x in ingredients: toks+=re.sub(r"[^a-z0-9\s]"," ",x.lower()).split()
        return " ".join(toks)

    def search(self, ingredients: list[str], k: int = 24) -> list[dict]:
        if not self.ok or not ingredients: return []
        qv=self.vectorizer.transform([self._ings_query(ingredients)])
        sims=(self.feats @ qv.T).toarray().ravel()
        order=np.argsort(-sims); out=[]; seen_title=set()
        for idx in order[:max(160,k)]:
            row=self.df.iloc[idx]
            title=str(row["title"]).strip()
            if title.lower() in seen_title: continue
            steps=[s.strip() for s in re.split(r"\r?\n|\|\|?|\.\s+(?=[A-Z])",str(row["steps"])) if s.strip()]
            ings=[s.strip() for s in re.split(r",|\|",str(row["ingredients"])) if s.strip()]
            img=str(row.get("image","")).strip()
            out.append({"title":title,"image":img,
                        "ingredients":ings,"steps":steps,"source":"local","score":float(sims[idx])})
            seen_title.add(title.lower())
            if len(out)==k: break
        return out

@st.cache_resource(show_spinner=False)
def _load_local_corpus()->LocalRecipeIndex:
    df=None; csvp=ASSETS_DIR/"recipes.csv"; parp=ASSETS_DIR/"recipes.parquet"
    try:
        if parp.exists(): df=pd.read_parquet(parp)
        elif csvp.exists(): df=pd.read_csv(csvp)
    except Exception: df=None
    return LocalRecipeIndex(df)
LOCAL_INDEX=_load_local_corpus()

# ---------------------- External APIs (optional) -----------------------
SPOON_KEY  = _get_env_or_secret("SPOONACULAR_KEY")

def _spoon_get(url: str, params: dict) -> dict | None:
    if not SPOON_KEY: return None
    try:
        p={"apiKey":SPOON_KEY}; p.update(params or {})
        r=requests.get(url, params=p, timeout=12, headers={"User-Agent":"SRF/1.0"})
        if r.status_code in (401,402,403): return None
        r.raise_for_status(); return r.json()
    except Exception: return None

@st.cache_data(show_spinner=False, ttl=900)
def fetch_spoonacular(ingredients: list[str], n: int, vegetarian: bool) -> list[dict]:
    if not ingredients or not SPOON_KEY: return []
    q=",".join([i.strip() for i in ingredients if i.strip()])
    url="https://api.spoonacular.com/recipes/complexSearch"
    params={"includeIngredients":q,"fillIngredients":True,"addRecipeInformation":True,
            "instructionsRequired":True,"number":max(18,n),"sort":"max-used-ingredients","ranking":2}
    if vegetarian: params["diet"]="vegetarian"
    js=_spoon_get(url, params)
    if not js or "results" not in js: return []
    out=[]
    for rec in js["results"]:
        title=(rec.get("title") or "").strip()
        image=(rec.get("image") or "").strip()
        ingr=[(it.get("original") or it.get("name","")).strip()
              for it in (rec.get("extendedIngredients") or []) if (it.get("original") or it.get("name",""))]
        steps=[]
        for block in (rec.get("analyzedInstructions") or []):
            for stp in (block.get("steps") or []):
                txt=(stp.get("step") or "").strip()
                if txt: steps.append(txt)
        if vegetarian and not _is_vegetarian(ingr):  # hard guard
            continue
        if title and ingr:
            out.append({"title":title,"image":image,"ingredients":ingr,"steps":steps,"source":"spoon","score":0.7})
        if len(out)==n: break
    return out

@st.cache_data(show_spinner=False, ttl=900)
def fetch_mealdb(ingredients: list[str], n: int, vegetarian: bool) -> list[dict]:
    try:
        seeds=[i for i in (ingredients[:3] or ["tomato"])]
        pool=[]
        for seed in seeds:
            js=requests.get("https://www.themealdb.com/api/json/v1/1/filter.php",
                            params={"i":seed},timeout=8, headers={"User-Agent":"SRF/1.0"}).json()
            for m in (js.get("meals") or [])[:10]:
                pool.append((m["idMeal"], m.get("strMeal","").strip(), (m.get("strMealThumb","") or "").strip()))
        out=[]
        for mid, title, img in pool[:24]:
            det=requests.get("https://www.themealdb.com/api/json/v1/1/lookup.php",
                             params={"i":mid},timeout=8, headers={"User-Agent":"SRF/1.0"}).json()
            meal=(det.get("meals") or [None])[0]
            if not meal: continue
            ings=[]
            for k in range(1,21):
                nm=(meal.get(f"strIngredient{k}") or "").strip()
                ms=(meal.get(f"strMeasure{k}") or "").strip()
                if nm: ings.append(f"{ms} {nm}".strip())
            ins=(meal.get("strInstructions") or "").strip()
            steps=[s.strip() for s in re.split(r"\r?\n|\.\s+(?=[A-Z])",ins) if s.strip()] if ins else []
            if vegetarian and not _is_vegetarian(ings):  # guard
                continue
            cand={"title":title,"image":img,"ingredients":ings,"steps":steps,"source":"mealdb","score":0.5}
            out.append(cand)
        return out[:n]
    except Exception: return []
def fetch_mealdb_seafood(n=25):
    out = []
    try:
        url = "https://www.themealdb.com/api/json/v1/1/filter.php?c=Seafood"
        r = requests.get(url, timeout=8)
        data = r.json().get("meals") or []
        for m in data[:n]:
            out.append({
                "title": m.get("strMeal",""),
                "image": m.get("strMealThumb",""),
                "ingredients": ["seafood","fish"],  # placeholder
                "steps": [],
                "source": "mealdb_seafood",
                "score": 0.55
            })
    except Exception:
        pass
    return out


# ------------------------- Plot helpers (Plotly) -----------------------
def bar_colored(x_labels, y_vals, title_y="", height=300):
    bars=[go.Bar(name=str(x), x=[x], y=[y]) for x,y in zip(x_labels,y_vals)]
    fig=go.Figure(data=bars)
    fig.update_layout(barmode="group", margin=dict(l=0,r=0,t=10,b=0),
                      yaxis=dict(title=title_y, rangemode="tozero"),
                      xaxis=dict(title=""), height=height, template="simple_white", showlegend=False)
    return fig

def kcal_line(df_days, df_kcal):
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df_days,y=df_kcal,mode="lines+markers",name="kcal"))
    fig.update_layout(height=320, template="simple_white", margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(title="kcal"))
    return fig

def macros_stacked(df_days, df_prot, df_fat, df_carb):
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df_days,y=df_prot,name="Protein"))
    fig.add_trace(go.Bar(x=df_days,y=df_fat, name="Fat"))
    fig.add_trace(go.Bar(x=df_days,y=df_carb,name="Carbs"))
    fig.update_layout(barmode="stack",height=320,template="simple_white",
                      margin=dict(l=0,r=0,t=10,b=0),yaxis=dict(title="grams"))
    return fig

# ----------------------------- PDF utils -------------------------------
def recipe_pdf(
    title: str,
    ingredients: List[str],
    recipe_text: str,
    nutrition: Dict[str, float],
    image_url: str
) -> Optional[bytes]:
    """Build a well-formatted, single-file PDF for a recipe.

    - Graceful image handling (aspect-preserving, network-safe).
    - Bulleted ingredients; numbered instructions.
    - Compact, readable styles and a clean nutrition table.
    - Page numbers in the footer.
    """
    if not PDF_OK:
        return None
    try:
        import io
        import requests
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
            ListFlowable, ListItem, KeepTogether
        )

        # --------------------- helpers ---------------------
        def _safe_text_lines(txt: str) -> list[str]:
            return [ln.strip() for ln in (txt or "").split("\n") if ln.strip()]

        def _fetch_image_bytes(url: str, timeout: float = 10.0) -> Optional[bytes]:
            if not url or not url.startswith(("http://", "https://")):
                return None
            try:
                r = requests.get(url, timeout=timeout, headers={"User-Agent": "SRF/1.0"})
                r.raise_for_status()
                return r.content
            except Exception:
                return None

        def _aspect_image(img_bytes: bytes, max_w: float, max_h: float) -> Optional[Image]:
            try:
                bio = io.BytesIO(img_bytes)
                im = Image(bio)
                iw, ih = im.wrap(0, 0)
                if iw <= 0 or ih <= 0:
                    return None
                scale = min(max_w / iw, max_h / ih, 1.0)
                im._restrictSize(iw * scale, ih * scale)
                return im
            except Exception:
                return None

        # --------------------- document ---------------------
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.65 * inch,
            title=title,
            author="Smart Recipe Finder (PRO)",
        )

        styles = getSampleStyleSheet()
        # Tweak defaults for a cleaner look
        styles["Title"].fontSize = 20
        styles["Title"].leading = 24
        styles["Heading2"].spaceBefore = 10
        styles["Heading2"].spaceAfter = 6

        body = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=10.8,
            leading=14.2,
            spaceAfter=3,
        )
        bullet_style = ParagraphStyle(
            "Bullet",
            parent=body,
            leftIndent=14,
            bulletIndent=6,
        )

        # Footer with page numbers
        def _footer(canvas, doc_):
            canvas.saveState()
            footer_txt = f"{eng_escape(title)} — page {doc_.page}"
            canvas.setFont("Helvetica", 9)
            canvas.setFillGray(0.4)
            canvas.drawRightString(
                doc_.pagesize[0] - doc_.rightMargin,
                0.45 * inch,
                footer_txt
            )
            canvas.restoreState()

        story: list = []

        # Title
        story.append(Paragraph(f"<b>{eng_escape(title)}</b>", styles["Title"]))
        story.append(Spacer(1, 8))

        # Optional hero image (scaled to fit)
        img_bytes = _fetch_image_bytes(image_url)
        if img_bytes:
            im = _aspect_image(img_bytes, max_w=6.2 * inch, max_h=3.5 * inch)
            if im:
                story.append(im)
                story.append(Spacer(1, 10))

        # Ingredients (bulleted)
        story.append(Paragraph("<b>Ingredients</b>", styles["Heading2"]))
        if ingredients:
            bullets = [
                ListItem(Paragraph(eng_escape(it), bullet_style), bulletText="•")
                for it in ingredients
            ]
            story.append(ListFlowable(bullets, bulletType="bullet", start="•", leftIndent=6))
        else:
            story.append(Paragraph("—", body))
        story.append(Spacer(1, 6))

        # Instructions (numbered)
        story.append(Paragraph("<b>Instructions</b>", styles["Heading2"]))
        steps = _safe_text_lines(recipe_text)
        if steps:
            numbered = [
                ListItem(Paragraph(eng_escape(s), body), value=i + 1)
                for i, s in enumerate(steps)
            ]
            story.append(ListFlowable(numbered, bulletType="1", start="1", leftIndent=6))
        else:
            story.append(Paragraph("—", body))
        story.append(Spacer(1, 10))

        # Nutrition table
        k = float(nutrition.get("kcal", 0.0) or 0.0)
        p = float(nutrition.get("protein", 0.0) or 0.0)
        f = float(nutrition.get("fat", 0.0) or 0.0)
        c = float(nutrition.get("carbs", 0.0) or 0.0)

        story.append(Paragraph("<b>Nutrition (approx.)</b>", styles["Heading2"]))
        data = [
            ["Calories (kcal)", f"{k:.1f}"],
            ["Protein (g)",     f"{p:.1f}"],
            ["Fat (g)",         f"{f:.1f}"],
            ["Carbs (g)",       f"{c:.1f}"],
        ]
        tbl = Table(data, colWidths=[2.2 * inch, 1.2 * inch])
        tbl.setStyle(TableStyle([
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.black),
            ("BACKGROUND",  (0, 0), (-1, 0), colors.whitesmoke),
            ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#333333")),
            ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.Color(1,1,1), colors.Color(0.98,0.98,0.98)]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",(0, 0), (-1, -1), 6),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(KeepTogether(tbl))

        # Build with footers
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buf.getvalue()

    except Exception:
        return None

def eng_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ----------------------------- Translation & TTS -----------------------
def translate_if(lines: list[str], lang_code: str) -> list[str]:
    if lang_code == "en" or not TRANS_OK: return lines
    try:
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source="auto", target=lang_code)
        return [tr.translate(t) for t in lines]
    except Exception:
        return lines

def _change_speed(seg, factor: float):
    new_rate = int(seg.frame_rate * factor)
    return seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate}).set_frame_rate(seg.frame_rate)
def _shift_pitch(seg, semitones: float):
    factor = 2.0 ** (semitones / 12.0)
    return _change_speed(seg, factor)

VOICE = {
    "Neutral": {"speed": 1.00, "pitch": 0.0},
    "Female-warm": {"speed": 0.98, "pitch": +1.0},
    "Female-light": {"speed": 1.03, "pitch": +2.0},
    "Male-soft": {"speed": 0.99, "pitch": -1.0},
    "Male-deep": {"speed": 0.96, "pitch": -6.0},   # deeper & slower
    "Fast": {"speed": 1.08, "pitch": 0.0},
    "Calm": {"speed": 0.95, "pitch": 0.0},
}

def tts_instructions(lines: List[str], lang_code: str, voice_preset: Dict[str, float], speed_factor: float) -> Optional[bytes]:
    if not TTS_OK: return None
    from gtts import gTTS
    from pydub import AudioSegment
    vp = dict(voice_preset or VOICE["Neutral"]); vp["speed"] = float(speed_factor)
    full = AudioSegment.silent(duration=250)
    for text in lines:
        bio = io.BytesIO(); gTTS(text=text, lang=lang_code).write_to_fp(bio); bio.seek(0)
        seg = AudioSegment.from_file(bio, format="mp3")
        seg = _shift_pitch(seg, vp.get("pitch", 0.0)); seg = _change_speed(seg, vp.get("speed", 1.0))
        full += seg + AudioSegment.silent(duration=140)
    out = io.BytesIO(); full.export(out, format="mp3"); return out.getvalue()

def podcast_script(title: str,
                   ingredients: list[str],
                   steps: list[str],
                   host_name: str,
                   chef_name: str) -> list[str]:
    """
    Host↔Chef podcast script with discourse connectors instead of numeric steps.
    - Uses all steps.
    - If the recipe supplies a single long paragraph, split it to pseudo-steps.
    - Keeps early Host prompts concise; Chef narrates instructions.
    """
    def _clean(s: str) -> str:
        s = " ".join((s or "").strip().split())
        while ".." in s:
            s = s.replace("..", ".")
        return s

    title = _clean(title)
    ingredients = [_clean(x) for x in (ingredients or []) if _clean(x)]
    raw_steps = [ _clean(x) for x in (steps or []) if _clean(x) ]

    # If the recipe provided only one blob, try to split into sentences/clauses.
    if len(raw_steps) <= 1:
        blob = raw_steps[0] if raw_steps else ""
        # split on sentence boundaries or semicolons/bullets
        chunks = []
        for piece in re.split(r"(?:\.\s+|\n+|;|\u2022|\u25CF)", blob):
            p = _clean(piece)
            if len(p) >= 4:
                chunks.append(p)
        if chunks:
            raw_steps = chunks

    # Connector sequence (last one is reserved for the final step)
    connectors = ["First,", "Then,", "Next,", "After that,", "Now,", "Meanwhile,", "Afterwards,"]
    final_connector = "Finally,"

    lead = [
        f"{host_name}: Welcome to Calm Kitchen. Today we're cooking {title}.",
        f"{host_name}: I'm joined by our chef, {chef_name}. Tell us about this dish.",
        f"{chef_name}: Thanks. This recipe highlights {', '.join(ingredients[:3]) if ingredients else 'simple pantry staples'}.",
        f"{host_name}: Great. Let's walk through the method."
    ]

    script = list(lead)

    n = len(raw_steps)
    for i, s in enumerate(raw_steps, start=1):
        if i == n:
            prefix = final_connector
        else:
            prefix = connectors[(i-1) % len(connectors)]
        script.append(f"{chef_name}: {prefix} {s}")

        # light pacing prompts early on only
        if i == 1:
            script.append(f"{host_name}: Nice start. What comes next?")
        elif i == 2:
            script.append(f"{host_name}: Understood—keep going.")

    script += [
        f"{host_name}: That wraps up {title}.",
        f"{chef_name}: Adjust seasoning to taste and serve.",
        f"{host_name}: Thanks for listening to Calm Kitchen."
    ]
    return script


def tts_podcast(lines: list[str], lang_code: str,
                host_voice: Dict[str, float], chef_voice: Dict[str, float],
                host_speed: float, chef_speed: float) -> Optional[bytes]:
    if not TTS_OK:
        return None

    from gtts import gTTS
    from pydub import AudioSegment

    # Voice profiles
    host_v = dict(host_voice or VOICE.get("Neutral", {}))
    host_v["speed"] = float(host_speed)

    chef_v = dict(chef_voice or VOICE.get("Neutral", {}))
    chef_v["speed"] = float(chef_speed)

    final_audio = AudioSegment.silent(duration=250)

    for ln in lines:
        text = str(ln or "").strip()
        if not text:
            continue

        # Safe dialog parsing: only split at the FIRST “: ”
        who, sep, say = text.partition(": ")
        if sep:  
            voice = chef_v if "chef" in who.lower() else host_v
            utter = say
        else:
            voice = host_v
            utter = text

        if not utter:
            continue

        # TTS (with fallback if language fails)
        bio = io.BytesIO()
        try:
            gTTS(text=utter, lang=lang_code).write_to_fp(bio)
        except Exception:
            bio = io.BytesIO()
            gTTS(text=utter, lang="en").write_to_fp(bio)  # fallback
        bio.seek(0)

        seg = AudioSegment.from_file(bio, format="mp3")
        seg = _shift_pitch(seg, float(voice.get("pitch", 0.0)))
        seg = _change_speed(seg, float(voice.get("speed", 1.0)))

        final_audio += seg + AudioSegment.silent(duration=140)

    out = io.BytesIO()
    final_audio.export(out, format="mp3")
    return out.getvalue()


def podcast_pdf(title: str, transcript_lines: list[str], lang_code: str = "en") -> Optional[bytes]:
    if not PDF_OK:
        return None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()

        # Localized PDF title
        full_title = f"{eng_escape(title)} {_lbl(lang_code, 'podcast_title_suffix')}"

        story = [
            Paragraph(f"<b>{full_title}</b>", styles["Title"]),
            Spacer(1, 12),
        ]

        for ln in transcript_lines:
            story.append(Paragraph(eng_escape(str(ln)), styles["Normal"]))
            story.append(Spacer(1, 4))

        doc.build(story)
        return buf.getvalue()

    except Exception:
        return None



# --------------------- Coverage + distinct selection -------------------
# ---- Protein/sweet heuristics for Non-veg filtering ----

# ---- Protein vocab & helpers (drop-in replacement) --------------------
PROTEIN_TOKENS = {
    # Generic seafood
    "fish","seafood","fish fillet","white fish","saltfish","smoked fish",
    "fresh fish","fried fish","baked fish","grilled fish",

    # Specific fishes
    "salmon","salmon fillet","tuna","tuna steak","cod","cod fillet",
    "haddock","pollock","tilapia","catfish","herring","anchovy","anchovies",
    "sardine","mackerel","trout","bass","sea bass","red snapper","snapper",
    "swordfish","eel",

    # Molluscs
    "squid","calamari","octopus","cuttlefish",

    # Shellfish
    "oyster","oysters","clam","clams","mussel","mussels","scallop","scallops",

    # Crustaceans
    "shrimp","shrimps","prawn","prawns","king prawns",
    "crab","crab meat","lobster","lobster tail","krill",

    # Land proteins
    "chicken","chicken breast","beef","pork","lamb","turkey","duck","veal","goat",

    # Eggs (special-cased elsewhere if needed)
    "egg","eggs",
}

# Canonical buckets used by the non-veg filter/boost
SEAFOOD_SET = {
    "fish","seafood","fish fillet","white fish","saltfish","smoked fish",
    "fresh fish","fried fish","baked fish","grilled fish",
    "salmon","salmon fillet","tuna","tuna steak","cod","cod fillet",
    "haddock","pollock","tilapia","catfish","herring","anchovy","anchovies",
    "sardine","mackerel","trout","bass","sea bass","red snapper","snapper",
    "swordfish","eel","squid","calamari","octopus","cuttlefish","oyster","oysters",
    "clam","clams","mussel","mussels","scallop","scallops","shrimp","shrimps",
    "prawn","prawns","king prawns","crab","crab meat","lobster","lobster tail","krill",
}
LAND_SET = {"chicken","chicken breast","beef","pork","lamb","turkey","duck","veal","goat"}
GENERIC_SEAFOOD = {"fish","seafood","fish fillet","white fish","fresh fish"}

# Helpful aliases (normalize user intent to canon tokens)
PROTEIN_ALIASES = {

    # Beef
    "steak": "beef",
    "ground beef": "beef",
    "minced beef": "beef",
    "beef mince": "beef",
    "sirloin": "beef",
    "tenderloin": "beef",
    "ribeye": "beef",
    "short rib": "beef",
    "brisket": "beef",
    "roast beef": "beef",
    "mince": "beef",     # heuristic

    # Pork
    "ground pork": "pork",
    "minced pork": "pork",
    "pork chop": "pork",
    "pork loin": "pork",
    "pork belly": "pork",
    "ham": "pork",
    "bacon": "pork",

    # Chicken
    "drumstick": "chicken",
    "thigh": "chicken",
    "breast": "chicken",
    "chicken breast": "chicken",
    "chicken thigh": "chicken",
    "chicken wing": "chicken",
    "wing": "chicken",
    "wings": "chicken",
    "ground chicken": "chicken",
    "minced chicken": "chicken",

    # Turkey
    "turkey breast": "turkey",
    "ground turkey": "turkey",
    "minced turkey": "turkey",

    # Lamb / Goat
    "lamb chop": "lamb",
    "lamb shank": "lamb",
    "ground lamb": "lamb",
    "minced lamb": "lamb",
    "mutton": "lamb",
    "goat": "lamb",
    
    # Generic fillet/filet → fish
    "fillet": "fish",
    "filet": "fish",
    "fish fillet": "fish",
    "fish filet": "fish",

    # General seafood terms
    "seafood": "whitefish",
    "fish": "fish",
    "fishes": "fish",
    "white fish": "whitefish",
    "whitefish": "whitefish",

    # Common fish types
    "salmon fillet": "salmon",
    "salmon steak": "salmon",
    "cod fillet": "cod",
    "haddock fillet": "haddock",
    "trout fillet": "trout",
    "tilapia fillet": "tilapia",
    "bass": "whitefish",
    "sea bass": "whitefish",
    "snapper": "whitefish",
    "red snapper": "whitefish",

    # Tuna variations
    "tuna steak": "tuna",
    "tuna fillet": "tuna",
    "canned tuna": "tuna",
    "tuna in water": "tuna",
    "tuna in oil": "tuna",

    # Oily fish
    "mackerel fillet": "mackerel",
    "sardine": "sardines",
    "sardines": "sardines",
    "anchovy": "anchovy",
    "anchovies": "anchovy",

    # Crustaceans
    "shrimp": "shrimp",
    "shrimps": "shrimp",
    "prawn": "shrimp",
    "prawns": "shrimp",
    "king prawn": "shrimp",
    "crab meat": "crab",
    "crab": "crab",
    "lobster tail": "lobster",
    "lobster": "lobster",

    # Molluscs
    "squid": "squid",
    "calamari": "squid",
    "octopus": "octopus",
    "cuttlefish": "squid",  # common grouping
    "scallop": "scallops",
    "scallops": "scallops",
    "clam": "clams",
    "clams": "clams",
    "mussel": "mussels",
    "mussels": "mussels",
}

SWEET_HINTS = {
    "dessert","pudding","cake","brownie","cookie","ice cream","whipped cream",
    "custard","sweet","sugar","syrup","strawberry","strawberries","banana","chocolate",
}

def _looks_sweet(title: str, ings: list[str]) -> bool:
    blob = (str(title) + " " + " ".join(ings)).lower()
    return any(k in blob for k in SWEET_HINTS)

def _accept_recipe_for_mode(rec_ings: list[str],
                            vegetarian: bool,
                            required_tokens: set[str] | None = None,
                            title: str = "") -> bool:
    # Diet gate
    if vegetarian and not _is_vegetarian(rec_ings):
        return False

    if not required_tokens:
        return True

    R = set(t for t in required_tokens if len(t) >= 2)
    have = _tokset(rec_ings) | _tokset([title])

    # If user asked for a protein, candidate must actually contain a protein
    req_has_protein = any(t in PROTEIN_TOKENS for t in R)
    rec_has_protein = any(t in PROTEIN_TOKENS for t in have)
    if req_has_protein and not rec_has_protein:
        return False

    # Minimal coverage: at least 60% of required tokens must be present
    covered = len(R & have)
    need = max(1, int(0.6 * len(R)))
    if covered < need:
        return False

    # If protein requested but the recipe looks like a dessert, reject
    if req_has_protein and _looks_sweet(title, rec_ings):
        return False

    return True


def _style_cycle_for(tokens_low: set[str]) -> list[str]:
    has_pasta = any(t in tokens_low for t in ["pasta","spaghetti","penne","tagliatelle"])
    has_rice  = any(t in tokens_low for t in ["rice","basmati","jasmine","risotto"])
    base = ["grill","skillet","curry","tray bake","one-pot pasta","pilaf","stir fry","soup","salad"]
    if has_pasta and "one-pot pasta" in base:
        base.remove("one-pot pasta"); base.insert(0,"one-pot pasta")
    if has_rice and "pilaf" in base:
        base.remove("pilaf"); base.insert(0,"pilaf")
    return base

def _inject_required_ingredients(rec: dict, required_tokens: set[str]) -> dict:
    """
    Strict-vegetarian mode:
    Ensure all required tokens appear in the ingredient list using canonical
    plant-based forms. Handles all dairy categories (milk, yogurt, butter,
    cheese, cream, ice cream), collapses stock tokens into 'vegetable stock',
    and maps gelatin to agar agar.
    """
    rec = dict(rec)
    if st.session_state.get("diet") != "Vegetarian":
        return rec

    # Current canonical tokens present
    have_tokens = _tokset(rec.get("ingredients", []))

    # Missing tokens (canonical)
    missing = [t for t in sorted(required_tokens) if t not in have_tokens]

    # Canonical dairy → plant-based equivalents
    DAIRY_CANON = {
        "milk":       "soy milk",
        "yogurt":     "soy yogurt",
        "yoghurt":    "soy yogurt",
        "butter":     "soy butter",
        "cheese":     "soy cheese",
        "cream":      "soy cream",
        "ice":        None,          # handled below (with 'ice cream')
        "icecream":   "soy ice cream",
        "ice_cream":  "soy ice cream",
    }

    # Additional explicit mappings
    EXTRA_CANON = {
        "gelatin": "agar agar",
    }

    # Helper: check if phrase is already in ingredient list
    def _has_phrase(phrase: str) -> bool:
        p = phrase.lower()
        for it in rec.get("ingredients", []):
            if p in str(it).lower():
                return True
        return False

    add_phrases: set[str] = set()

    # --- 1) Dairy injections ---
    for dairy_word, plant_equiv in DAIRY_CANON.items():
        if dairy_word in missing:
            # Handle ice cream explicitly
            if dairy_word in {"ice", "icecream", "ice_cream"}:
                if not _has_phrase("soy ice cream"):
                    add_phrases.add("soy ice cream")
                continue

            if plant_equiv is None:
                continue  # ignore bare "ice" or unhandled patterns

            if not _has_phrase(plant_equiv):
                add_phrases.add(plant_equiv)

    # --- 2) Gelatin → Agar agar ---
    for k, v in EXTRA_CANON.items():
        if k in missing and not _has_phrase(v):
            add_phrases.add(v)

    # --- 3) Stock tokens → vegetable stock ---
    if "stock" in missing or "vegetable" in missing:
        if not _has_phrase("vegetable stock"):
            add_phrases.add("vegetable stock")

    # --- 4) Avoid injecting lone plant markers ---
    bare_block = {"soy", "almond", "oat", "coconut", "vegan", "plant"}

    # Inject non-dairy, non-stock tokens
    for t in missing:
        if t in DAIRY_CANON:  # handled
            continue
        if t in EXTRA_CANON:
            continue
        if t in {"vegetable", "stock"}:
            continue
        if t in bare_block:
            continue
        if not _has_phrase(t):
            add_phrases.add(t)

    # --- 5) Write back injections ---
    if add_phrases:
        rec["ingredients"] = list(rec.get("ingredients", [])) + sorted(add_phrases)

        # Patch steps for clarity
        steps = list(rec.get("steps", []))
        patch = "Ensure you add the required items now: " + ", ".join(sorted(add_phrases)) + "."
        insert_at = 2 if len(steps) >= 2 else 1
        steps.insert(insert_at, patch)
        rec["steps"] = steps

    return rec


# ======================= NEW: deterministic fresh 3 =====================
def retrieve_strict_top3(ings: list[str], cuisines: list[str], vegetarian: bool, salt: str) -> list[dict]:
    """
    Assemble a diversified pool (local index + external APIs), enforce required-token
    coverage and vegetarian rules, synthesize steps/images as needed, and return
    TOPK distinct recipes (distinct by title+steps).
    """
    import hashlib, random  # local-only deps

    TOPK_LOCAL = int(globals().get("TOPK", 3))
    salted = salt or "seed"

    # --- Local distinct selector (title+steps signatures) ----------------------
    def _norm_local(s: str) -> str:
        return (s or "").strip().lower()

    def _distinct_topk_force_local(cands: list[dict], topk: int) -> list[dict]:
        # rank by score (desc), then greedily keep first unseen (title+steps)
        cands = sorted(cands, key=lambda z: float(z.get("score", 0.0)), reverse=True)
        seen_title: set[str] = set()
        seen_steps: set[str] = set()
        out: list[dict] = []
        for r in cands:
            title_norm = _norm_local(r.get("title", ""))
            steps_norm = "\n".join(_norm_local(s) for s in r.get("steps", []) if isinstance(s, str))[:4096]
            tkey = hashlib.sha1(title_norm.encode("utf-8")).hexdigest()
            skey = hashlib.sha1(steps_norm.encode("utf-8")).hexdigest()
            if tkey in seen_title or skey in seen_steps:
                continue
            r.setdefault("image", "")
            r.setdefault("ingredients", [])
            r.setdefault("steps", [])
            seen_title.add(tkey)
            seen_steps.add(skey)
            out.append(r)
            if len(out) == topk:
                break
        return out
    # -------------------------------------------------------------------------

    # --- Protein intent expansion (non-veg only) ------------------------------
    LAND_SET = {
        "meat", "red meat", "white meat",
        "chicken", "chicken breast", "chicken thigh", "drumstick", "wings", "ground chicken", "minced chicken",
        "beef", "steak", "ribeye", "sirloin", "tenderloin", "ground beef", "minced beef", "roast beef",
        "lamb", "mutton", "ground lamb", "minced lamb",
        "goat", "chevon",
        "pork", "bacon", "ham", "prosciutto", "pancetta", "sausage",
        "veal",
        "turkey", "ground turkey", "turkey breast",
        "duck", "duck breast", "duck leg"
    }
    SEAFOOD_SET = {
        "seafood", "fish",
        "salmon", "tuna", "cod", "haddock", "sea bass", "tilapia", "halibut", "trout", "mackerel",
        "shrimp", "prawn", "prawns", "scallop", "scallops",
        "mussel", "mussels", "clam", "clams", "octopus", "squid", "calamari",
        "crab", "lobster", "anchovy", "anchovies", "sardine", "sardines"
    }
    PROTEIN_TOKENS_LOCAL = LAND_SET | SEAFOOD_SET

    MEAT_SYNONYMS    = {"meat", "red meat", "white meat"}
    SEAFOOD_SYNONYMS = {"seafood", "fish", "fishes", "fish meat", "white fish", "shellfish"}
    CHICKEN_SYNS     = {"chicken", "chicken meat", "chicken breast", "chicken thigh", "drumstick", "wings", "ground chicken", "minced chicken"}
    BEEF_SYNS        = {"beef", "steak", "ribeye", "sirloin", "tenderloin", "ground beef", "minced beef", "roast beef"}
    LAMB_SYNS        = {"lamb", "mutton", "ground lamb", "minced lamb"}
    GOAT_SYNS        = {"goat", "chevon"}
    PORK_SYNS        = {"pork", "bacon", "ham", "prosciutto", "pancetta", "sausage"}
    VEAL_SYNS        = {"veal"}
    TURKEY_SYNS      = {"turkey", "turkey meat", "ground turkey", "turkey breast"}
    DUCK_SYNS        = {"duck", "duck breast", "duck leg"}

    def _norm_token(s: str) -> str:
        s = (s or "").strip().lower()
        if s == "lamp":  # common typo
            return "lamb"
        return s

    def _has_any(blob: str, syns: set[str]) -> bool:
        return any(k in blob for k in syns)

    def _expand_requested_proteins(req_proteins: set[str], user_ings: list[str]) -> set[str]:
        corpus = [*user_ings, *list(req_proteins)]
        blob = " ".join(_norm_token(x) for x in corpus)
        out = set(_norm_token(x) for x in req_proteins if x)

        # generic
        if _has_any(blob, MEAT_SYNONYMS):     out |= LAND_SET
        if _has_any(blob, SEAFOOD_SYNONYMS):  out |= SEAFOOD_SET

        # specific land families
        if _has_any(blob, CHICKEN_SYNS):  out |= CHICKEN_SYNS
        if _has_any(blob, BEEF_SYNS):     out |= BEEF_SYNS
        if _has_any(blob, LAMB_SYNS):     out |= LAMB_SYNS
        if _has_any(blob, GOAT_SYNS):     out |= GOAT_SYNS
        if _has_any(blob, PORK_SYNS):     out |= PORK_SYNS
        if _has_any(blob, VEAL_SYNS):     out |= VEAL_SYNS
        if _has_any(blob, TURKEY_SYNS):   out |= TURKEY_SYNS
        if _has_any(blob, DUCK_SYNS):     out |= DUCK_SYNS

        # literal group tokens
        if any(t in out for t in {"meat", "red meat", "white meat"}): out |= LAND_SET
        if any(t in out for t in {"seafood", "fish"}):                out |= SEAFOOD_SET
        return out
    # -------------------------------------------------------------------------

    # 1) Required tokens (veg-stripped if needed) + record what was dropped
    req_tokens, dropped, req_proteins = _required_tokens_for_mode(ings, vegetarian)
    st.session_state["veg_dropped"] = dropped if vegetarian else set()
    pool: list[dict] = []

    # Expanded protein intents (non-veg only)
    req_proteins_expanded: set[str] = set()
    if not vegetarian:
        req_proteins_expanded = _expand_requested_proteins(req_proteins, ings)

    # 2) Local results first (bigger pool → better diversity)
    if LOCAL_INDEX.ok:
        query_local = list(req_tokens) if vegetarian else ings
        for r in LOCAL_INDEX.search(query_local, k=220):
            if not _accept_recipe_for_mode(
                r.get("ingredients", []),
                vegetarian,
                required_tokens=req_tokens,
                title=r.get("title", "")
            ):
                continue
            rr = dict(r)
            rr["score"] = 0.64 + 0.36 * float(r.get("score", 0.0))
            pool.append(rr)

    # 3) External APIs (veg uses veg-stripped terms)
    query_terms = list(req_tokens) if vegetarian else ings
    pool += fetch_spoonacular(query_terms, n=36, vegetarian=vegetarian)
    pool += fetch_mealdb(query_terms,     n=30, vegetarian=vegetarian)
    if not vegetarian:
        pool += fetch_mealdb_seafood(n=30)

    # 4) Resolve images + synthesize steps if missing
    tokens_low  = _tokset(query_terms)
    styles_pref = _style_cycle_for(tokens_low)
    seen_imgs: set[str] = set()

    new_pool: list[dict] = []
    for idx, r in enumerate(pool):
        rr = dict(r)  # work on a copy

        # ---- STRICT VEGETARIAN PATCH: dairy/stock → plant, with UI note
        if vegetarian:
            rr = _apply_veg_substitutions_recipe(rr, salted)
            rr, swapped = _veg_milk_swap_in_recipe(rr)
            rr = _normalize_veg_dairy_terms(rr)
            if swapped:
                prev = st.session_state.get("veg_dropped", set())
                st.session_state["veg_dropped"] = prev | {f"{s} → soy/veg-alt" for s in swapped}

        # ---- NON-VEG: honor expanded protein intents
        if not vegetarian and req_proteins_expanded:
            have = _tokset(rr.get("ingredients", [])) | _tokset([rr.get("title", "")])
            if not any(p in have for p in req_proteins_expanded):
                continue  # skip this candidate

        # ---- Image resolve/refresh (salted for diversity)
        rr["image"] = _resolve_image(
            rr.get("image"),
            vegetarian,
            rr.get("title", ""),
            f"{salted}#{idx}",
            seen_imgs
        )

        # ---- Fill steps when absent (deterministic but varied per candidate)
        if not rr.get("steps"):
            style = styles_pref[idx % len(styles_pref)] if styles_pref else "skillet"
            rr["steps"] = distinct_steps_from_ingredients(
                rr.get("title", ""),
                rr.get("ingredients", []),
                style=style,
                seed=1000 + idx,
                required_tokens=req_tokens
            )

        # ---- Non-veg boost when an actual protein exists (except eggs)
        if not vegetarian:
            have = _tokset(rr.get("ingredients", [])) | _tokset([rr.get("title", "")])
            if any(t in PROTEIN_TOKENS_LOCAL for t in have if t not in {"egg", "eggs"}):
                rr["score"] = float(rr.get("score", 0.0)) + 0.20

        new_pool.append(rr)

    # replace pool with filtered/enhanced candidates
    pool = new_pool

    # 5) Enforce required tokens into every candidate
    pool = [_inject_required_ingredients(r, req_tokens) for r in pool]

    # 6) Hard vegetarian guard immediately before ranking
    if vegetarian:
        pool = [r for r in pool if _is_vegetarian(r.get("ingredients", []))]

    # 7) Select distinct top-k (by title+steps)
    uniq = _distinct_topk_force_local(pool, TOPK_LOCAL)

    # 8) If fewer than TOPK, synthesize DISTINCT fallbacks (salted → per-search variety)
    if len(uniq) < TOPK_LOCAL:
        rng = random.Random(int(hashlib.sha1(("syn" + salted).encode()).hexdigest(), 16))

        veg_templates = [
            ("One-Pot Pasta Primavera", ["olive oil","garlic","tomato","basil","pasta","spinach","nut parmesan"], "one-pot pasta"),
            ("Chickpea & Spinach Curry", ["oil","onion","garlic","ginger","chickpeas","tomato","curry powder","soy yogurt"], "curry"),
            ("Mediterranean Orzo Salad", ["orzo","olive oil","lemon","tomato","cucumber","vegan feta","olive","parsley"], "salad"),
            ("Mushroom Barley Soup", ["olive oil","onion","garlic","mushroom","barley","thyme","vegetable stock"], "soup"),
            ("Tofu Stir-Fry", ["tofu","soy sauce","ginger","garlic","broccoli","bell pepper","sesame oil"], "stir fry"),
            ("Tray-Bake Veg Medley", ["olive oil","broccoli","cauliflower","potato","paprika","lemon"], "tray bake"),
            ("Lentil Pilaf", ["olive oil","onion","garlic","brown lentils","rice","cumin","bay leaf"], "pilaf"),
            ("Grilled Tofu & Veg Skewers", ["marinated firm tofu","zucchini","bell pepper","red onion","olive oil","lemon","oregano"], "grill"),
            ("Smoky Eggplant Kebab", ["eggplant","olive oil","garlic","lemon","parsley","sumac","flatbread"], "grill"),
            ("Tandoori Tofu Tikka", ["tofu (pressed)","soy yogurt","garam masala","ginger","garlic","lemon"], "grill"),
        ]

        nonveg_templates = [
            ("Chicken Tomato Skillet", ["olive oil","garlic","onion","chicken breast","tomato","basil"], "skillet"),
            ("Beef Pilaf", ["oil","onion","garlic","ground beef","rice","bay leaf","parsley"], "pilaf"),
            ("Seafood Curry", ["oil","onion","garlic","ginger","shrimp","curry powder","coconut milk","rice"], "curry"),
            ("Lemon Herb Baked Salmon", ["salmon","olive oil","lemon","garlic","parsley"], "tray bake"),
            ("Turkey Noodle Soup", ["turkey","noodle","carrot","celery","onion","stock"], "soup"),
            ("Prawn Stir-Fry", ["shrimp","soy sauce","ginger","garlic","snap peas","sesame oil"], "stir fry"),
            ("Beef & Veggie Ragu", ["olive oil","onion","garlic","ground beef","tomato","oregano","pasta"], "one-pot pasta"),
            ("Chicken Shish Kebab", ["chicken thigh","olive oil","garlic","lemon","paprika","oregano","red onion"], "grill"),
            ("Adana Kebab (Turkish)", ["ground beef","ground lamb","red pepper flakes","sumac","parsley","flatbread"], "grill"),
            ("Tandoori Chicken", ["chicken","yogurt","tandoori masala","ginger","garlic","lemon"], "grill"),
            ("Grilled Salmon with Dill", ["salmon","olive oil","lemon","dill","garlic"], "grill"),
        ]
        T = veg_templates if vegetarian else nonveg_templates
        req_tokens_list = list(req_tokens)

        seen_fallback_imgs: set[str] = set()
        for idx in range(40):
            name, base_ings, style = rng.choice(T)
            chosen = list(base_ings)

            # inject required tokens deterministically but varied
            for tok in sorted(req_tokens_list):
                if tok not in _tokset(chosen):
                    chosen.insert(1 + (idx % 2), tok)

            title = f"{name} with {_best3(ings)}" if ings else name
            steps = distinct_steps_from_ingredients(
                title, chosen, style=style, seed=300 + idx + len(uniq), required_tokens=req_tokens
            )

            # salted fallback image to keep results unique even without APIs
            img = _pexels_image(f"{title} plated recipe", per_page=14, page=1 + (idx % 3))
            if not img:
                img = _fallback_food_image(vegetarian, title, f"{salted}|{idx}", seen_fallback_imgs)
            if img in seen_fallback_imgs:
                img = _fallback_food_image(vegetarian, title + str(idx), f"{salted}|{idx*7}", seen_fallback_imgs)
            seen_fallback_imgs.add(img)

            uniq.append({
                "title": title,
                "image": img,
                "ingredients": chosen,
                "steps": steps,
                "source": "synthetic",
                "score": 0.20 - 0.001 * idx
            })
            uniq = _distinct_topk_force_local(uniq, TOPK_LOCAL)
            if len(uniq) == TOPK_LOCAL:
                break

    # 9) Final pass: ensure one unique image per recipe and enforce tokens again
    out: list[dict] = []
    seen_final: set[str] = set()
    for j, r in enumerate(uniq[:TOPK_LOCAL]):
        if vegetarian and not _is_vegetarian(r.get("ingredients", [])):  # safety
            continue
        r["image"] = _resolve_image(
            r.get("image"),
            vegetarian,
            r.get("title", ""),
            f"{salted}::final{j}",
            seen_final
        )
        out.append(_inject_required_ingredients(r, req_tokens))

    return out[:TOPK_LOCAL]







# =====================================================================
# UI
# =====================================================================
left, right = st.columns([1.55, 1.0])

with left:
    st.subheader("Ingredients")
    demo = "tomato, basil, garlic, olive oil"
    st.session_state["diet"] = st.radio("Diet", ["Non-vegetarian","Vegetarian"], horizontal=True, index=0, key="diet_radio")
    st.text_area("Comma-separated or one per line", value=demo, height=120,
                 placeholder="e.g., tomato, basil, garlic, olive oil", key="ing_text")

    if st.button("Search recipes", type="primary", key="btn_search"):
        ings, sig = _canon_ings(st.session_state.get("ing_text",""))
        if not ings: 
            st.warning("Please provide at least one ingredient."); 
            st.stop()

        # --- NEW: swap dairy milk -> soy/almond in Vegetarian mode BEFORE prediction/search ---
        veg_now = (st.session_state["diet"] == "Vegetarian")
        if veg_now:
            ings, _swapped = _veg_milk_swap(ings)
            if _swapped:
                prev = st.session_state.get("veg_dropped", set())
                note = {f"{s} → soy/almond milk" for s in _swapped}
                st.session_state["veg_dropped"] = set(prev) | note
        # -------------------------------------------------------------------

        cuisines, probs = predict_topk(pipe, INV, ings, k=TOPK)
        meta = {c: {"title": f"{c.title()}-Style"} for c in cuisines}

        is_veg = (st.session_state["diet"] == "Vegetarian")
        # salt ensures different photos/recipes across searches even with similar input
        salt = uuid.uuid4().hex[:8] + "|" + str(int(time.time()))
        recs = retrieve_strict_top3(ings, cuisines, vegetarian=is_veg, salt=salt)

        # map titles -> ingredients for planner macros
        title_to_ings = {r["title"]: r["ingredients"] for r in recs}

        st.session_state.update({
            "pred_ready": True,
            "df_pred": pd.DataFrame({"cuisine": cuisines, "probability": probs}),
            "cuisines": cuisines, "meta": meta, "selected": cuisines[0] if cuisines else None,
            "ings": ings, "last_sig": sig, "recs_top3": recs, "search_salt": salt,
            "title_to_ings": title_to_ings,
            "podcast_context": {"title": recs[0]["title"] if recs else "",
                                "ingredients": recs[0]["ingredients"] if recs else [],
                                "steps": recs[0]["steps"] if recs else []}
        })
        st.rerun()


with left:
    if st.session_state["pred_ready"] == True:
        df_pred  = st.session_state["df_pred"]
        diet_now = st.session_state["diet"]
        salt_key = st.session_state.get("search_salt","")

        tab_pred, tab_pod, tab_plan = st.tabs(["🔮 Predictions", "🎙️ Podcast", "📅 Planner"])

        # -------------------- Predictions --------------------
        with tab_pred:
            st.markdown(f"### Top predictions · **{diet_now}**")
            if not df_pred.empty:
                st.plotly_chart(
                    bar_colored(df_pred["cuisine"], df_pred["probability"], title_y="Probability", height=180),
                    use_container_width=True, config={"displayModeBar": False}, key=f"pred_chart_{salt_key}"
                )
            if diet_now == "Vegetarian":
                dropped = st.session_state.get("veg_dropped", set())
                if dropped:
                    st.info("Removed for strict-veg: " + ", ".join(sorted(set(dropped))))

            # Icons (avoid f-strings with backslashes)
            icon_veg = ""
            if ICON_VEG.exists():  icon_veg = "![](" + str(ICON_VEG).replace("\\","/") + ")"
            icon_non = ""
            if ICON_NONV.exists(): icon_non = "![](" + str(ICON_NONV).replace("\\","/") + ")"

            st.markdown("### Recommended recipes (unique per search)")
            recs = st.session_state.get("recs_top3", [])
            for i, rec in enumerate(recs, 1):
                badge = f"{icon_veg} " if diet_now=="Vegetarian" else f"{icon_non} "
                st.markdown(f'<div class="pill">{badge}{rec["title"]}  <span class="caption">source: {rec.get("source","")}</span></div>',
                            unsafe_allow_html=True)
                st.image(rec["image"], width=380)  # explicit width -> no use_column_width warning

                with st.expander("Ingredients", expanded=False):
                    st.write("\n".join(f"- {x}" for x in rec["ingredients"]))
                with st.expander("Steps", expanded=True):
                    st.write("\n".join(f"{k+1}. {s}" for k, s in enumerate(rec["steps"])))

                m = macro_totals(rec["ingredients"])
                st.plotly_chart(
                    bar_colored(["Calories (kcal)","Protein (g)","Fat (g)","Carbs (g)"],
                                [m["kcal"], m["protein"], m["fat"], m["carbs"]], title_y="Amount"),
                    use_container_width=True, config={"displayModeBar": False}, key=f"macro_{i}_{salt_key}"
                )

                # PDF (language choice is for the PDF text only; UI stays English)
                col_pdf, col_lang, col_v, col_s, col_btn = st.columns([1,1,1,1,1])
                with col_pdf:
                    if PDF_OK:
                        st.caption(" ")
                with col_lang:
                    lang_ui = st.selectbox(f"Speech language #{i}", list(LANGUAGE_CHOICES.keys()),
                                           index=0, key=f"lang_{i}")
                with col_v:
                    default_idx = list(VOICE.keys()).index(["Female-warm","Male-soft","Male-deep"][(i-1)%3])
                    preset = st.selectbox(f"Voice preset #{i}", list(VOICE.keys()),
                                          index=default_idx, key=f"voice_{i}")
                with col_s:
                    spd = st.slider(f"Speed #{i}", min_value=0.85, max_value=1.20,
                                    value=float(VOICE[preset]["speed"]), step=0.01, key=f"speed_{i}")
                with col_btn:
                    if st.button("🎧 Instruction Voice", key=f"audio_{i}", use_container_width=True):
                        code = LANGUAGE_CHOICES[lang_ui]
                        # Do not mutate displayed steps; translate a copy only for audio
                        lines_src = [f"Today's dish is: {rec['title']}.", "Here are the instructions."] + \
                                [f"Step {k+1}: {s}" for k, s in enumerate(rec["steps"])]
                        lines = translate_if(lines_src, code)
                        audio = tts_instructions(lines, code, VOICE[preset], spd)
                        if audio:
                            st.audio(audio, format="audio/mp3")
                            st.download_button("⬇ MP3", audio,
                                               file_name=f"{i}_{LANGUAGE_CHOICES[lang_ui]}_instruction.mp3",
                                               mime="audio/mpeg", use_container_width=True, key=f"mp3_{i}")

                # PDF download (translated or not, but source data unchanged)
                code = LANGUAGE_CHOICES[lang_ui]
                pdf_text_lines = [f"{k+1}. {s}" for k, s in enumerate(rec["steps"])]
                pdf_lines = translate_if(pdf_text_lines, LANGUAGE_CHOICES[lang_ui])
                pdfb = recipe_pdf(rec["title"], rec["ingredients"], "\n".join(pdf_lines), m, rec.get("image",""))
                if pdfb:
                    safe = re.sub(r"[^a-zA-Z0-9]+","_",rec["title"]).lower()
                    st.download_button("📄 Download PDF", data=pdfb, file_name=f"{safe}.pdf",
                                       mime="application/pdf", use_container_width=True, key=f"pdf_{i}_{salt_key}")

        # ------------------------ PODCAST ------------------------
        with tab_pod:
            st.caption("Interview-style podcast: Host ↔ Chef.")
            recs = st.session_state.get("recs_top3", [])
            opts = [f"{i+1}) {r['title']}" for i, r in enumerate(recs)]
            pick = st.selectbox("Select recipe", list(range(len(opts))),
                                format_func=lambda i: opts[i], index=0 if opts else 0,
                                key=f"pod_pick_{salt_key}") if opts else 0
            sel = recs[pick] if recs else {"title":"","ingredients":[],"steps":[]}
            c1, c2 = st.columns(2)
            with c1:
                host = st.text_input("Host name", value="Host", key=f"pod_host_{salt_key}")
                v_h  = st.selectbox("Host voice", list(VOICE.keys()),
                                    index=list(VOICE.keys()).index("Male-soft"), key=f"pod_vh_{salt_key}")
                s_h  = st.slider("Host speed", 0.85, 1.20, float(VOICE[v_h]["speed"]), 0.01, key=f"pod_sh_{salt_key}")
            with c2:
                chef = st.text_input("Chef name", value="Chef", key=f"pod_chef_{salt_key}")
                v_c  = st.selectbox("Chef voice", list(VOICE.keys()),
                                    index=list(VOICE.keys()).index("Male-deep"), key=f"pod_vc_{salt_key}")
                s_c  = st.slider("Chef speed", 0.85, 1.20, float(VOICE[v_c]["speed"]), 0.01, key=f"pod_sc_{salt_key}")
            lang_ui = st.selectbox("Podcast language", list(LANGUAGE_CHOICES.keys()), index=0, key=f"pod_lang_{salt_key}")
            code = LANGUAGE_CHOICES[lang_ui]
            lines = translate_if(podcast_script(sel["title"], sel["ingredients"], sel["steps"], host, chef), code)
            st.text_area("Transcript (preview)", "\n".join(lines), height=180, key=f"pod_tx_{salt_key}")
            colp1, colp2, _ = st.columns([1,1,1])
            with colp1:
                if st.button("🎙️ Generate Podcast MP3", key=f"pod_build_{salt_key}"):
                    audio = tts_podcast(lines, code, VOICE[v_h], VOICE[v_c], s_h, s_c)
                    if audio:
                        st.audio(audio, format="audio/mp3")
                        st.download_button("⬇ Download Podcast MP3", audio,
                                           file_name=f"podcast_{re.sub(r'[^a-zA-Z0-9]+','_',sel['title']).lower()}.mp3",
                                           mime="audio/mpeg", use_container_width=True, key=f"pod_dl_{salt_key}")
            with colp2:
                code = LANGUAGE_CHOICES[lang_ui]
                title_tr = translate_if([sel["title"]], code)[0]
                pdfp = podcast_pdf(title_tr, lines, lang_code=code)
                
                if pdfp:
                    st.download_button("📄 Download Transcript PDF", pdfp,
                                       file_name=f"podcast_{re.sub(r'[^a-zA-Z0-9]+','_',sel['title']).lower()}_transcript.pdf",
                                       mime="application/pdf", use_container_width=True, key=f"pod_pdf_{salt_key}")

        # ------------------------ Planner ------------------------
        with tab_plan:
            st.caption("Weekly planner + charts.")
            mode = st.radio("Add mode", ["From top-3","Custom"], horizontal=True, key=f"plan_mode_{salt_key}")
            if mode == "From top-3":
                opts = [f"{i+1}) {r['title']}" for i, r in enumerate(st.session_state.get("recs_top3", []))]
                pick = st.selectbox("Pick recipe", opts, index=0 if opts else None, key=f"plan_pick_{salt_key}")
                dish = (st.session_state.get("recs_top3", [])[int(pick.split(')')[0])-1]["title"]) if opts else ""
            else:
                dish = st.text_input("Recipe name (custom)", value="", key=f"plan_custom_{salt_key}")
            days = [f"Day {i}" for i in range(1,8)]
            cL, cR = st.columns([1,2])
            with cL: dsel = st.selectbox("Day", days, index=0, key=f"plan_day_{salt_key}")
            with cR:
                a,b,c = st.columns(3)
                with a:
                    if st.button("Add/Update", use_container_width=True, key=f"plan_add_{salt_key}"):
                        if dish.strip():
                            found=False
                            for r in st.session_state["meal_plan"]:
                                if r["Day"]==dsel: r["Recipe"]=dish.strip(); found=True; break
                            if not found:
                                nxt=(max([r["Order"] for r in st.session_state["meal_plan"]] or [0])+1)
                                st.session_state["meal_plan"].append({"Order":nxt,"Day":dsel,"Recipe":dish.strip()})
                with b:
                    if st.button("Delete day", use_container_width=True, key=f"plan_del_{salt_key}"):
                        st.session_state["meal_plan"] = [r for r in st.session_state["meal_plan"] if r["Day"]!=dsel]
                with c:
                    if st.button("Clear all", use_container_width=True, key=f"plan_clear_{salt_key}"):
                        st.session_state["meal_plan"] = []
            if st.session_state["meal_plan"]:
                df = pd.DataFrame(st.session_state["meal_plan"]).sort_values("Order").reset_index(drop=True)
                # compute macros from stored ingredients where available
                t2i = st.session_state.get("title_to_ings", {})
                totals = []
                for _, r in df.iterrows():
                    ings_for = t2i.get(r["Recipe"], [r["Recipe"]])  # fallback uses name
                    totals.append(macro_totals(ings_for))
                mdf = pd.DataFrame(totals); df = pd.concat([df, mdf], axis=1)
                st.dataframe(df, use_container_width=True, height=260)
                st.plotly_chart(kcal_line(df["Day"], df["kcal"]),  use_container_width=True, config={"displayModeBar": False}, key=f"kcal_fig_{salt_key}")
                st.plotly_chart(macros_stacked(df["Day"], df["protein"], df["fat"], df["carbs"]),
                                use_container_width=True, config={"displayModeBar": False}, key=f"mac_fig_{salt_key}")
                W = {"kcal":float(df["kcal"].sum()),"protein":float(df["protein"].sum()),
                     "fat":float(df["fat"].sum()),"carbs":float(df["carbs"].sum())}
                st.info(f"Weekly totals — kcal: {W['kcal']:.0f}, Protein: {W['protein']:.1f} g, Fat: {W['fat']:.1f} g, Carbs: {W['carbs']:.1f} g")

with right:
    st.subheader("How to use")
    st.markdown(
        """
        <ul class="howto">
          <li><b>Enter ingredients</b> — comma-separated or one per line.</li>
          <li><b>Select diet</b> — Vegetarian strictly excludes meat, seafood, egg, gelatin and stock derivatives.</li>
          <li>Click <b>Search recipes</b> — every search returns <b>three new, distinct recipes</b> with
              <b>unique images</b>. Veg mode uses only vegetarian candidates.</li>
          <li>Open a recipe to review <b>ingredients</b> and <b>steps</b>; download a <b>PDF</b> or generate
              <b>instruction audio</b> in your chosen language (on-screen text stays in English).</li>
          <li>Use <b>Podcast</b> for a host ↔ chef dialogue with adjustable voices and speed.</li>
          <li>Plan your week in <b>Planner</b>; macro totals and charts update automatically.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# =====================================================================
# End of file
# ===================================================================== 
