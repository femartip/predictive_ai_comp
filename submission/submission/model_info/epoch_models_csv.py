import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# Copy of scripts/preprocess/epoch_models_csv.py for the submission, which runs it offline on model_info/data. Two
# changes: ROOT falls back to this folder (no pyproject.toml in the submission), and huggingface_hub is imported only
# where it is used (build_subjects, not called here).
import pandas as pd

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
            Path(__file__).resolve().parent)
URL = "https://epoch.ai/data/ai_models.zip"
ZIP = ROOT / "data" / "ai_models.zip"
SOURCE = "all_ai_models.csv"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "models_raw.csv"
MODELS = PROCESSED / "models.csv"
MODELS_DEV_URL = "https://models.dev/api.json"  # one request, the whole catalogue; no key
MODELS_DEV_CACHE = ROOT / "data" / "models_dev.json"
RELEASED_URL = "https://www.releasedmodels.com/api/v1/models"  # paginated, max 100 per page, 60 requests/min per IP; no key
RELEASED_CACHE = ROOT / "data" / "releasedmodels.json"
BUILD_REPO = "aims-foundations/measurement-db"  # the benchmark build: subjects tables with provider and release_date
HF_API = "https://huggingface.co"  # anonymous access only (no token): 500 API requests per 5 minutes per IP
HF_CACHE = ROOT / "data" / "hf_hub.json"

DROP_COLUMNS = ["Link", "Citations", "Abstract", "WikiText and Penn Treebank data"]
NUMERIC = ["Parameters", "Training compute (FLOP)", "Training dataset size (total)", "Training time (hours)",
           "Finetune compute (FLOP)", "Post-training compute (FLOP)", "Hardware quantity", "Epochs", "Batch size",
           "Training compute cost (2023 USD)", "Training power draw (W)"]

# Epoch fields kept in models.csv (the ones analysed in scripts/analysis/epoch_models.ipynb) → output column.
FIELDS = {
    "Publication date": "publication_date",
    "Organization": "organization",
    "Country (of organization)": "country",
    "Model accessibility": "model_accessibility",
    "Domain": "epoch_domain",  # not "domain": the benchmark files have their own domain column
    "Base model": "base_model",
    "Parameters": "parameters",
    "Training compute (FLOP)": "training_compute_flop",
    "Training dataset size (total)": "training_dataset_size",
    "Finetune compute (FLOP)": "finetune_compute_flop",
    "Post-training compute (FLOP)": "post_training_compute_flop",
    "Training hardware": "training_hardware",
    "Frontier model": "frontier_model",
    "Confidence": "confidence",
}

# Benchmark entries that are not a single identifiable model.
NOT_A_MODEL = {"Test": "test submission", "Update Eval": "test submission",
               "KISS-1": "unidentified leaderboard entry", "Magellanes": "unidentified leaderboard entry"}

# normalized_name → (Epoch model, reason), for names Epoch spells differently.
ALIASES = {
    "OpenAI GPT-4": ("GPT-4 Turbo (Apr 2024)",
                     "only subject is mmdocrag 'gpt-4-turbo'; the API alias points to gpt-4-turbo-2024-04-09"),
    "Mistral Small 24B": ("Mistral Small 3", "Mistral Small 3 is the 24B model released Jan 2025"),
    "OpenAI CUA": ("Computer-Using Agent (CUA)", "same system, Epoch spells the name out"),
    "Stanford s1.1 32B": ("s1.1", "Epoch lists s1.1 without the size"),
}
SUFFIX = re.compile(r"^(a\d+b|instruct|it|chat|preview|exp|experimental|latest|\d{4}|\d{6})+$")


# ---------------------------------------------------------------- download and clean

def download():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:
        ZIP.write_bytes(r.read())
    with zipfile.ZipFile(ZIP) as z:
        print(f"downloaded {URL} ({ZIP.stat().st_size / 1e6:.1f} MB) → {ZIP.relative_to(ROOT)}: {z.namelist()}")


def to_number(v):
    """Numeric value of a cell; a number repeated with commas counts as that number, anything else is NaN."""
    if pd.isna(v):
        return float("nan")
    try:
        values = {float(p) for p in str(v).split(",")}
    except ValueError:
        return float("nan")
    return values.pop() if len(values) == 1 else float("nan")


def clean():
    with zipfile.ZipFile(ZIP) as z, z.open(SOURCE) as f:
        df = pd.read_csv(f, low_memory=False)
    for c in NUMERIC:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before = df[c].notna().sum()
            df[c] = df[c].map(to_number)
            print(f"  {c}: {before - df[c].notna().sum()} non-numeric values set to empty")
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    assert df.Model.is_unique, "duplicate model names"
    out = df.drop(columns=DROP_COLUMNS).assign(**{"Publication date": df["Publication date"].dt.date})
    out.to_csv(RAW, index=False)
    print(f"{len(out):,} models × {out.shape[1]} columns → {RAW.relative_to(ROOT)}")
    return df  # with the dropped columns: the abstract is read for context_window and thinking


# ---------------------------------------------------------------- models in our benchmarks

def build_subjects(benchmark):
    """provider and release_date per subject_id from the build's subjects table, at the revision the benchmark
    script used (its manifest); the processed files carry model identity only."""
    from huggingface_hub import hf_hub_download
    manifest = json.loads((PROCESSED / f"{benchmark}.manifest.json").read_text(encoding="utf-8"))
    path = hf_hub_download(BUILD_REPO, f"{benchmark}/formatted_tables/subjects.parquet", repo_type="dataset",
                           revision=manifest["source_revision"])
    return pd.read_parquet(path, columns=["subject_id", "provider", "release_date"])


def benchmark_models():
    """One row per normalized_name across data/processed/<benchmark>.jsonl."""
    fields = ["subject_id", "model_name", "normalized_name", "thinking"]
    rows = []
    for path in sorted(PROCESSED.glob("*.jsonl")):
        seen = {}
        with open(path, encoding="utf-8") as f:  # streamed: some files hold long prompts
            for line in f:
                row = json.loads(line)
                seen.setdefault(row["subject_id"], {k: row[k] for k in fields})
        build = build_subjects(path.stem).set_index("subject_id")
        rows += [r | {"benchmark": path.stem, "provider": build.provider.get(r["subject_id"]),
                      "release_date": build.release_date.get(r["subject_id"])} for r in seen.values()]
    assert rows, f"no <benchmark>.jsonl in {PROCESSED}: run the benchmark scripts first"
    subj = pd.DataFrame(rows)
    models = (subj.groupby("normalized_name")
              .agg(provider=("provider", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                   release_date=("release_date", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
                   benchmarks=("benchmark", lambda s: ";".join(sorted(set(s)))),
                   display_names=("model_name", lambda s: ";".join(sorted(set(s)))),
                   n_subjects=("subject_id", "size"),
                   # Did any run think / were all runs with a known value non-thinking? Read by fill_capabilities.
                   runs_thinking=("thinking", lambda s: True if s.eq(True).any() else (False if s.eq(False).any() else None)))
              .reset_index())
    models["kind"] = [("agent system" if n.startswith("Agent: ") else NOT_A_MODEL.get(n, "model"))
                      for n in models.normalized_name]
    print(f"{len(subj)} subjects in {subj.benchmark.nunique()} benchmarks → {len(models)} normalized names "
          f"({(models.kind == 'model').sum()} models)")
    return models


# ---------------------------------------------------------------- matching

def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def name_variants(name, provider, providers):
    """The name itself and the name without a provider prefix (the row's own, or any provider in the data)."""
    names = {name}
    for p in ([provider] if isinstance(provider, str) else []) + providers:
        if name.lower().startswith(p.lower() + " "):
            names.add(name[len(p):].strip())
            break
    return names


def candidates(names, normalized_name, epoch):
    if normalized_name in ALIASES:
        return "alias", epoch[epoch.Model == ALIASES[normalized_name][0]]
    keys = {key(n) for n in names}
    for stage, col in [("exact", "key"), ("exact, date dropped", "key_nodate")]:
        c = epoch[epoch[col].isin(keys)]
        if len(c):
            return stage, c
    c = epoch[[any(k2.startswith(k) and SUFFIX.match(k2[len(k):]) for k in keys) for k2 in epoch.key_nodate]]
    return ("prefix", c) if len(c) else ("none", c)


def pick(cands, names, release_date):
    """Most similar name, then the publication date closest to the build's release date."""
    rel = pd.to_datetime(release_date, errors="coerce")
    sim = cands.Model.map(lambda m: max(difflib.SequenceMatcher(None, n.lower(), m.lower()).ratio() for n in names))
    gap = (cands["Publication date"] - rel).abs().dt.days if pd.notna(rel) else pd.Series(0, index=cands.index)
    return cands.assign(_sim=sim.round(2), _gap=gap.fillna(1e6)).sort_values(["_sim", "_gap"], ascending=[False, True]).index[0]


def is_open(row):
    acc, flag = row["Model accessibility"], row["Open model weights?"]
    if isinstance(acc, str):
        return acc.startswith("Open weights")
    return {"Yes": True, "No": False}.get(flag)


# ---------------------------------------------------------------- sizes written in the name

SIZE = r"(\d+(?:\.\d+)?)([bm])"  # 7b, 1.5b, 135m
TOKEN_TOTAL = re.compile(rf"^{SIZE}$")                 # 480b
TOKEN_ACTIVE = re.compile(rf"^a{SIZE}$")               # a35b: active parameters of a mixture of experts
TOKEN_X = re.compile(rf"^(\d+)x{SIZE}$")               # 8x7b: experts × size of one expert
TOKEN_EXPERTS = re.compile(r"^(\d+)e$")                # 128e: number of experts (Llama 4: "17b-128e")


def to_count(value, unit):
    return float(value) * (1e9 if unit == "b" else 1e6)


def size_from_name(name):
    """Total and active parameters and number of experts written in a model name; None where absent.

    "Qwen3 235B A22B" → 235e9 total, 22e9 active · "Mixtral 8x7B" → 8 experts (total not in the name) ·
    "llama4-mave-17b-128e" → 128 experts, and with an expert count the plain size is the active size, as in
    Meta's "Llama-4-Maverick-17B-128E".
    """
    out = {"total": None, "active": None, "experts": None}
    if not isinstance(name, str):
        return out
    sizes = []
    for t in re.split(r"[\s\-_/]+", name.lower()):
        if m := TOKEN_ACTIVE.match(t):
            out["active"] = to_count(*m.groups())
        elif m := TOKEN_X.match(t):
            out["experts"] = int(m[1])
        elif m := TOKEN_EXPERTS.match(t):
            out["experts"] = int(m[1])
        elif m := TOKEN_TOTAL.match(t):
            sizes.append(to_count(*m.groups()))
    if len(sizes) == 1:  # several plain sizes are ambiguous: leave them
        if out["experts"] is not None and out["active"] is None and not TOKEN_X.search(name.lower()):
            out["active"] = sizes[0]
        else:
            out["total"] = sizes[0]
    return out


# Epoch's `Parameters notes` state MoE in words ("Mixture-of-Experts", "MoE", "256 experts") or by giving an active
# count ("671B total, 37B active"), which only sparse models have. Speculative notes do not count.
NOTE_MOE = re.compile(r"mixture[- ]of[- ]experts|\bmoe\b|\bexperts?\b|\bactiv(?:e|ated)\b", re.I)
NOTE_SPECULATIVE = re.compile(r"suspect|assum|guess", re.I)
NOTE_ACTIVE = re.compile(r"(\d+(?:\.\d+)?)\s*(b|billion|t|trillion)\b[a-z\s-]{0,15}?\bactiv", re.I)  # no number in between
NOTE_EXPERTS = re.compile(r"number of experts:\s*(\d+)|(\d+)\s+experts\b", re.I)


def size_from_notes(notes):
    """MoE, active parameters and number of experts stated in Epoch's parameter notes; None where absent."""
    out = {"moe": None, "active": None, "experts": None}
    if not isinstance(notes, str) or NOTE_SPECULATIVE.search(notes) or not NOTE_MOE.search(notes):
        return out
    out["moe"] = True
    if m := NOTE_ACTIVE.search(notes):
        out["active"] = float(m[1]) * (1e12 if m[2].lower() in ("t", "trillion") else 1e9)
    if m := NOTE_EXPERTS.search(notes):
        out["experts"] = int(m[1] or m[2])
    return out


def fill_sizes(df):
    """parameters where Epoch has none, plus active_parameters, moe and n_experts, from the names and Epoch's notes."""
    found = []
    for r in df.itertuples():
        names = [r.normalized_name, *str(r.display_names or "").split(";"), r.epoch_model]
        sizes = [size_from_name(n) for n in names]
        found.append({k: next((s[k] for s in sizes if s[k] is not None), None) for k in ("total", "active", "experts")})
    f = pd.DataFrame(found, index=df.index)
    notes = pd.DataFrame([size_from_notes(n) for n in df.parameters_notes], index=df.index)
    df["parameters_source"] = df.parameters.notna().map({True: "epoch", False: None})
    fill = df.parameters.isna() & f.total.notna()
    df.loc[fill, "parameters"] = f.total[fill].astype(float)   # f.total is all None (object) when no name has a size
    df.loc[fill, "parameters_source"] = "name"
    df["active_parameters"] = f.active.fillna(notes.active)
    df["active_parameters_source"] = f.active.notna().map({True: "name", False: None}).where(
        f.active.notna(), notes.active.notna().map({True: "epoch notes", False: None}))
    df["n_experts"] = f.experts.fillna(notes.experts).astype("Int64")
    df["n_experts_source"] = f.experts.notna().map({True: "name", False: None}).where(
        f.experts.notna(), notes.experts.notna().map({True: "epoch notes", False: None}))
    # MoE only with evidence (name first, then Epoch's notes); without it the model is not known to be dense.
    by_name = f.active.notna() | f.experts.notna()
    by_notes = ~by_name & notes.moe.eq(True)
    df["moe"] = (by_name | by_notes).map({True: True, False: None})
    df["moe_source"] = by_name.map({True: "name", False: None}).where(by_name, by_notes.map({True: "epoch notes", False: None}))
    print(f"  parameters: filled {fill.sum()} from the name; still empty: {df[df.kind == 'model'].parameters.isna().sum()} "
          f"models · MoE: {by_name.sum()} from the name, {by_notes.sum()} more from Epoch's notes · "
          f"active parameters known: {df.active_parameters.notna().sum()} · experts known: {df.n_experts.notna().sum()}")
    return df


def match(models, epoch):
    assert all(v[0] in set(epoch.Model) for v in ALIASES.values()), "alias target missing from Epoch"
    epoch = epoch.assign(key=epoch.Model.map(key),
                         key_nodate=epoch.Model.str.replace(r"\s*\([^)]*\)", "", regex=True).map(key))
    providers = sorted(set(models.provider.dropna()), key=len, reverse=True)
    out = []
    for r in models.itertuples():
        row = {"match_stage": "not a model", "n_candidates": 0, "epoch_model": None}
        e = None
        if r.kind == "model":
            names = name_variants(r.normalized_name, r.provider, providers)
            stage, cands = candidates(names, r.normalized_name, epoch)
            row |= {"match_stage": stage, "n_candidates": len(cands)}
            if len(cands):
                e = epoch.loc[pick(cands, names, r.release_date)]
                row["epoch_model"] = e.Model
        row["open_weights"] = is_open(e) if e is not None else None
        row |= {out_col: (e[col] if e is not None else None) for col, out_col in FIELDS.items()}
        row["parameters_notes"] = e["Parameters notes"] if e is not None else None  # read by fill_sizes, not written
        row["abstract"] = e["Abstract"] if e is not None else None  # read by fill_capabilities, not written
        row["hf_developer"] = e["Hugging Face developer id"] if e is not None else None  # read by fill_hf, not written
        out.append(row)
    df = pd.concat([models.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    df["publication_date"] = pd.to_datetime(df.publication_date).dt.date
    df["open_weights_source"] = df.open_weights.notna().map({True: "epoch", False: None})
    # Fill what the benchmark files leave empty. provider: the name's own prefix when it is a provider used
    # elsewhere in the data ("Zhipu GLM-4.5" → Zhipu), else Epoch's first organization ("Google,Google DeepMind"
    # → Google). release_date: Epoch's publication date.
    empty = df.provider.isna()
    from_name = df.normalized_name.map(lambda n: next((p for p in providers if n.startswith(p + " ")), None))
    from_epoch = df.organization.str.split(",").str[0].str.strip()
    df.loc[empty, "provider"] = from_name[empty].fillna(from_epoch[empty])
    print(f"  provider: filled {empty.sum() - df.provider.isna().sum()} empty values "
          f"({(empty & from_name.notna()).sum()} from the name, the rest from Epoch); still empty: {df.provider.isna().sum()}")
    df["release_date_source"] = df.release_date.notna().map({True: "benchmark", False: None})
    empty = df.release_date.isna() & df.publication_date.notna()
    df.loc[empty, "release_date"] = df.loc[empty, "publication_date"].astype(str)
    df.loc[empty, "release_date_source"] = "epoch"
    print(f"  release_date: filled {empty.sum()} empty values from Epoch; still empty: {df.release_date.isna().sum()}")
    return df


# ---------------------------------------------------------------- capabilities from Epoch's domain and text

INPUT_DOMAINS = {"Vision", "Multimodal", "Video", "Audio", "Speech"}
# Vision markers in model names (for models Epoch lacks or whose domain says only Language).
VISION_NAME = re.compile(r"\bvl\b|vl\d|internvl|\bqvq\b|vision|omni|minicpm-o|\d(?:\.\d+)?v\b", re.I)  # 4.5V: version + V

# Context window stated in Epoch's abstract or parameter notes. K and M are read as 1,024 and 1,024² tokens, as in
# the exact figures the text gives ("256K" = 262,144; "1,048,576 token context window").
CTX_WORDS = {"one": 1, "two": 2, "ten": 10}
CTX_UNIT = {"k": 1024, "thousand": 1024, "m": 1024 ** 2, "million": 1024 ** 2}
CTX_QTY = r"(\d[\d,]*(?:\.\d+)?|one|two|ten)[\s-]*(k|m|million|thousand)?\b"
CTX_AFTER = re.compile(CTX_QTY + r"(?=[\s-]*(?:tokens?\b|context\b))", re.I)  # "128K context", "2M token"
CTX_BEFORE = re.compile(r"context(?: length| window)?s?\b[^.\d]{0,25}?" + CTX_QTY, re.I)  # "context length: 32,768"
CTX_FROM_TO = re.compile(r"from\s+" + CTX_QTY + r"[^.\d]{0,10}to\s+" + CTX_QTY, re.I)  # "expanded from 128K to 200K"
# A sentence naming a variant the model does not have is about a sibling ("Llama 4 Scout offers … 10M").
CTX_VARIANTS = {"scout", "maverick", "behemoth", "pro", "flash", "lite", "nano", "mini", "max", "plus", "turbo", "air"}

# Thinking: name markers and clear statements in Epoch's abstract.
THINK_NAME = re.compile(r"think|reason|\br1\b|distill|\bqwq\b|\bqvq\b|\bo[134]\b|\bo[134]-", re.I)
THINK_TEXT = re.compile(r"reasoning model|thinking model|hybrid reasoning|thinking mode|extended,? (?:step-by-step )?"
                        r"thinking|think(?:s|ing)? before (?:responding|answering|acting)|reasoning and non-reasoning", re.I)


def context_window(text, names):
    """Native context window in tokens stated in the text; None if absent. The first statement wins."""
    if not isinstance(text, str):
        return None
    own = {w for n in names if isinstance(n, str) for w in re.split(r"[\W_]+", n.lower())}
    qty = lambda num, unit: int((CTX_WORDS.get(num.lower()) or float(num.replace(",", "")))
                                * CTX_UNIT.get((unit or "").lower(), 1))
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if "context" not in sentence.lower() or (set(re.split(r"[\W_]+", sentence.lower())) & CTX_VARIANTS) - own:
            continue
        if m := CTX_FROM_TO.search(sentence):
            return qty(m[3], m[4])
        hits = [h for p in (CTX_BEFORE, CTX_AFTER) for m in p.finditer(sentence) if (h := qty(*m.groups())) >= 1000]
        if hits:
            return min(hits) if "native" in sentence.lower() else hits[0]  # native size, not the extended one
    return None


def fill_capabilities(df):
    """multimodal_input from Epoch's domain; context_window and thinking from names, runs and Epoch's text."""
    # multimodal_input: true with evidence (Epoch's domain, or a vision marker in a name), otherwise empty. A domain
    # of only "Language" does not prove text-only input: Epoch lists e.g. Claude Opus 4.7 and Qwen3.5 that way.
    by_domain = df.epoch_domain.map(lambda d: isinstance(d, str) and bool(set(d.split(",")) & INPUT_DOMAINS))
    names = df.normalized_name + ";" + df.display_names.fillna("") + ";" + df.epoch_model.fillna("")
    by_vname = ~by_domain & names.str.contains(VISION_NAME)
    df["multimodal_input"] = (by_domain | by_vname).map({True: True, False: None})
    df["multimodal_input_source"] = by_domain.map({True: "epoch domain", False: None}).where(by_domain, by_vname.map({True: "name", False: None}))

    text = df.parameters_notes.fillna("") + "\n" + df.abstract.fillna("")
    df["context_window"] = [context_window(t, [r.normalized_name, *str(r.display_names).split(";"), r.epoch_model])
                            for t, r in zip(text, df.itertuples())]
    df["context_window"] = df.context_window.astype("Int64")
    df["context_window_source"] = df.context_window.notna().map({True: "epoch text", False: None})

    # thinking: does the model have a thinking/reasoning mode? true if any source says so; false only if every run
    # with a known value was non-thinking and nothing else says the model can think; otherwise empty.
    by_runs, by_name, by_text = df.runs_thinking.eq(True), names.str.contains(THINK_NAME), df.abstract.fillna("").str.contains(THINK_TEXT)
    df["thinking"] = None
    df["thinking_source"] = None
    for mask, source in [(by_text, "epoch abstract"), (by_name, "name"), (by_runs, "runs")]:  # last one wins the label
        df.loc[mask, "thinking"] = True
        df.loc[mask, "thinking_source"] = source
    no = df.runs_thinking.eq(False) & df.thinking.isna()
    df.loc[no, "thinking"] = False
    df.loc[no, "thinking_source"] = "runs"
    df.loc[df.kind != "model", ["multimodal_input", "multimodal_input_source", "context_window", "context_window_source", "thinking",
                                 "thinking_source"]] = None
    print(f"  multimodal_input: {df.multimodal_input.eq(True).sum()} true ({df.multimodal_input_source.value_counts().to_dict()}) · "
          f"context_window: {df.context_window.notna().sum()} · thinking: {df.thinking.eq(True).sum()} true "
          f"({df.thinking_source[df.thinking.eq(True)].value_counts().to_dict()}), {df.thinking.eq(False).sum()} false")
    return df


# ---------------------------------------------------------------- external catalogues: models.dev, ReleasedModels

def fetch_json(url, retries=5):
    """GET a JSON document; on HTTP 429 wait for Retry-After (or back off) and retry."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "predictive-ai-comp/epoch_models_csv"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 2 ** (attempt + 1)))


def cached(path, offline, fetch):
    """Fetch and cache a catalogue; with --offline, or if fetching fails, reuse the cache."""
    if not offline:
        try:
            data = fetch()
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if not path.exists():
                raise
            print(f"  could not fetch {path.name} ({e}); using the cached copy")
    return json.loads(path.read_text(encoding="utf-8"))


def models_dev_records(offline):
    """One record per (provider, model) listing in models.dev."""
    data = cached(MODELS_DEV_CACHE, offline, lambda: fetch_json(MODELS_DEV_URL))
    rows = [{"provider_id": pid, "provider_name": p.get("name") or pid, **rec}
            for pid, p in data.items() for rec in (p.get("models") or {}).values()]
    print(f"  models.dev: {len(rows):,} listings from {len(data)} providers")
    return rows


def releasedmodels_records(offline):
    """All models from ReleasedModels' list endpoint, 100 per page, paced under its 60 requests/min limit."""
    def fetch():
        rows, page, pages = [], 1, 1
        while page <= pages:
            d = fetch_json(f"{RELEASED_URL}?per_page=100&page={page}")
            rows += d["data"]
            pages = d["meta"]["total_pages"]
            page += 1
            time.sleep(1.1)
        return rows
    rows = cached(RELEASED_CACHE, offline, fetch)
    print(f"  ReleasedModels: {len(rows)} models")
    return rows


FULL_DATE = re.compile(r"[-_ :@]?(?:20\d\d-?\d\d-?\d\d)$|\s*\([^)]*\)$|:(?:free|beta|exacto)$", re.I)
RUN_MARKS = re.compile(r"\((?:low|medium|high|xhigh|max|think|thinking)\)|[-_ ]no[-_ ]think", re.I)


def catalogue_keys(*names):
    """Keys of a catalogue listing: its name and id (last path segment), with and without a trailing full date."""
    out = set()
    for n in names:
        if isinstance(n, str) and n:
            n = n.split("/")[-1].split(":")[-1] if "/" in n else n
            out |= {key(n), key(FULL_DATE.sub("", n.strip()))}
    return {k for k in out if len(k) > 2}


def our_keys(r, providers):
    """Keys to look a model up by, most specific first: the benchmark display names without run markers ("(high)",
    "-no-think"), i.e. what was actually run ("gpt-4-turbo" behind "OpenAI GPT-4"); then the normalized name (with
    and without provider); then Epoch's name without its version date."""
    names = sorted({RUN_MARKS.sub("", n).strip() for n in str(r.display_names or "").split(";")})
    names += sorted(name_variants(r.normalized_name, r.provider, providers), key=len)
    if isinstance(r.epoch_model, str):
        names.append(re.sub(r"\s*\([^)]*\)", "", r.epoch_model))
    keys = []
    for n in names:
        k = key(n)
        if len(k) > 2 and k not in keys:
            keys.append(k)
    return keys


def lookup(df, records, id_of, name_of, provider_of, completeness, providers):
    """For each model, the catalogue record found by exact key. Among several: the model creator's own listing,
    then one whose id (not just its display name) matches, then the most complete. Hosting providers often reuse
    a plain name for a variant ("Grok 4 Fast" for grok-4-fast-non-reasoning), so the id is the safer evidence.
    Returns a list aligned with df (None where not found)."""
    index = {}
    for rec in records:
        by_id = catalogue_keys(id_of(rec))
        for k in by_id | catalogue_keys(name_of(rec)):
            index.setdefault(k, []).append((rec, k in by_id))
    out = []
    for r in df.itertuples():
        hits = [] if r.kind != "model" else next((index[k] for k in our_keys(r, providers) if k in index), [])
        creator = key(r.provider) if isinstance(r.provider, str) else None
        is_own = lambda rec: bool(creator) and any(key(p).startswith(creator) or creator.startswith(key(p))
                                                   for p in provider_of(rec) if key(p))
        ranked = sorted(hits, key=lambda h: (is_own(h[0]), h[1], completeness(h[0])), reverse=True)
        out.append({"best": ranked[0][0], "own": [h[0] for h in hits if is_own(h[0])], "all": [h[0] for h in hits]}
                   if ranked else None)
    return out


def field_value(found, get):
    """A field across all matching listings: the creator's own listing first, else the most common value."""
    if not found:
        return None
    own = [v for v in map(get, found["own"]) if v is not None]
    if own:
        return own[0]
    values = [v for v in map(get, found["all"]) if v is not None]
    return max(set(values), key=values.count) if values else None


def fill(df, col, values, source):
    """Fill the empty cells of `col` with `values` (aligned with df), recording the source. Returns how many."""
    values = pd.Series(values, index=df.index, dtype=object)
    if f"{col}_source" not in df:
        df[f"{col}_source"] = df[col].notna().map({True: "epoch", False: None})
    empty = df[col].isna() & values.notna() & df.kind.eq("model")
    df[col] = df[col].astype(object)
    df.loc[empty, col] = values[empty]
    df.loc[empty, f"{col}_source"] = source
    return int(empty.sum())


def fill_external(df, offline):
    """Fill what Epoch, the names and the runs leave empty: first from models.dev, then from ReleasedModels."""
    providers = sorted(set(df.provider.dropna()), key=len, reverse=True)
    df["architecture"] = df.moe.map({True: "moe"})
    df["architecture_source"] = df.moe_source.where(df.moe.eq(True))
    df["knowledge_cutoff"] = None
    df["knowledge_cutoff_source"] = None
    counts = {}

    # models.dev: release date, knowledge cutoff, reasoning, input modalities, open weights, context limit.
    md_records = models_dev_records(offline)
    # A provider that stamps one knowledge cutoff on many unrelated model families is using a placeholder.
    families = {}
    for x in md_records:
        families.setdefault((x["provider_id"], x.get("knowledge")), set()).add(x.get("family"))
    placeholder = {k for k, fam in families.items() if k[1] and len(fam) >= 5}
    knowledge = lambda x: None if (x["provider_id"], x.get("knowledge")) in placeholder else x.get("knowledge")
    md = lookup(df, md_records,
                id_of=lambda x: x.get("id"), name_of=lambda x: x.get("name"),
                provider_of=lambda x: [x["provider_id"], x["provider_name"]],
                completeness=lambda x: sum(v is not None for v in [x.get("knowledge"), x.get("release_date"),
                                                                    (x.get("limit") or {}).get("context")]),
                providers=providers)
    df["models_dev_id"] = [f"{x['best']['provider_id']}/{x['best']['id']}" if x else None for x in md]
    g = lambda f: [field_value(x, f) for x in md]
    inputs = g(lambda x: frozenset((x.get("modalities") or {}).get("input") or []) or None)
    counts["models.dev"] = {
        "release_date": fill(df, "release_date", g(lambda x: x.get("release_date")), "models.dev"),
        "knowledge_cutoff": fill(df, "knowledge_cutoff", g(knowledge), "models.dev"),
        "thinking": fill(df, "thinking", g(lambda x: x.get("reasoning")), "models.dev"),
        # image, video or audio input is multimodal; text (and pdf) only is not
        "multimodal_input": fill(df, "multimodal_input", [None if i is None else bool(i & {"image", "video", "audio"})
                                                          for i in inputs], "models.dev"),
        "open_weights": fill(df, "open_weights", g(lambda x: x.get("open_weights")), "models.dev"),
        "context_window": fill(df, "context_window", g(lambda x: (x.get("limit") or {}).get("context")), "models.dev"),
    }

    # ReleasedModels: architecture (dense/MoE) for open-weight models, sizes, context, release date, open weights.
    rm = lookup(df, releasedmodels_records(offline),
                id_of=lambda x: x.get("slug"), name_of=lambda x: x.get("name"),
                provider_of=lambda x: [(x.get("company") or {}).get("name") or "", (x.get("company") or {}).get("slug") or ""],
                completeness=lambda x: sum(x.get(f) is not None for f in ["params_total_b", "params_active_b", "context_window"]),
                providers=providers)
    df["releasedmodels_slug"] = [x["best"]["slug"] if x else None for x in rm]
    g = lambda f: [field_value(x, f) for x in rm]
    billions = lambda v: float(v) * 1e9 if v is not None else None
    counts["ReleasedModels"] = {
        "open_weights": fill(df, "open_weights", g(lambda x: x.get("open_weights")), "releasedmodels"),
        "release_date": fill(df, "release_date", g(lambda x: x.get("release_date")), "releasedmodels"),
        # Sizes only for open-weight models: for closed models they are guesses.
        "parameters": fill(df, "parameters", [v if ow is True else None for v, ow in
                                              zip(g(lambda x: billions(x.get("params_total_b"))), df.open_weights)], "releasedmodels"),
        "active_parameters": fill(df, "active_parameters", [v if ow is True else None for v, ow in
                                                            zip(g(lambda x: billions(x.get("params_active_b"))), df.open_weights)],
                                  "releasedmodels"),
        "context_window": fill(df, "context_window", g(lambda x: x.get("context_window")), "releasedmodels"),
        "multimodal_input": fill(df, "multimodal_input", g(lambda x: True if x.get("type") == "multimodal" else None),
                                 "releasedmodels"),
        # Labs of closed models do not disclose their architecture, so ReleasedModels' label is a guess there:
        # use it only for open-weight models, whose architecture is published with the weights.
        "architecture": fill(df, "architecture", [a if a != "unknown" and ow is True else None for a, ow in
                                                  zip(g(lambda x: x.get("architecture")), df.open_weights)],
                             "releasedmodels"),
    }
    # moe follows a known architecture: dense → false, moe → true.
    arch_moe = df.architecture.map({"moe": True, "dense": False})
    counts["ReleasedModels"]["moe"] = fill(df, "moe", arch_moe, "releasedmodels")
    df["context_window"] = pd.to_numeric(df.context_window).astype("Int64")
    for k in ["parameters", "active_parameters"]:
        df[k] = pd.to_numeric(df[k])
    for src, c in counts.items():
        print(f"  {src}: matched {(df['models_dev_id' if src == 'models.dev' else 'releasedmodels_slug'].notna()).sum()} "
              f"models · filled {c}")
    return df


# ---------------------------------------------------------------- Hugging Face Hub (open-weight models, no token)

# Repo name = model key + optional -Instruct/-it/-chat/-hf, a 0.x version (Mistral-7B-Instruct-v0.3) or a date.
# Not "-V2": that is a new model (K2-Think-V2).
# Size tokens are allowed too when our name has none ("Llama 4 Maverick" → Llama-4-Maverick-17B-128E-Instruct).
HF_SUFFIX = re.compile(r"^(instruct|it|chat|hf|preview|v0\d*|\d{4}|\d{6}|\d+(?:\d)?[bm]|a\d+b|\d+e)*$")
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TEXT_CONFIGS = ("text_config", "llm_config", "language_config")  # the language model inside a multimodal config


def hf_get(path, retries=5):
    """Anonymous GET on the Hub; None on 401/403/404 (gated or missing). Paced under 500 requests / 5 minutes."""
    time.sleep(0.7)
    try:
        return fetch_json(HF_API + path, retries=retries)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return None
        raise


def hf_repo(r, providers, provider_owners):
    """The model's repo: search by its names, keep repos whose name is one of its keys (optionally followed by
    -Instruct, -it, -hf, a 0.x version or a date) and whose owner is the model's developer, then take the most
    downloaded across all queries. Developer = Epoch's Hugging Face developer id (or the ids Epoch gives other
    models of the same provider), the provider, or the start of the model's name. This keeps third-party uploads
    (fine-tunes, quantizations, re-uploads) out."""
    keys = our_keys(r, providers)
    owners = {key(o) for o in [r.hf_developer, r.provider] if isinstance(o, str)}
    owners |= provider_owners.get(r.provider, set())
    words = r.normalized_name.split()
    owners |= {key(" ".join(words[:i])) for i in (1, 2) if len(words) > i}
    queries = []
    # The search matches substrings of repo ids, so also try "_" for "." (InternVL2_5, MiniCPM-o-2_6) and the
    # name's stem without its size or version ("MiniCPM-o").
    for n in [*name_variants(r.normalized_name, r.provider, providers), *str(r.display_names or "").split(";")]:
        q = RUN_MARKS.sub("", n).strip().replace(" ", "-")
        stem = re.split(r"-(?=\d)", q)[0]
        for v in (q, q.replace(".", "_"), stem):
            if len(key(v)) > 2 and v not in queries:
                queries.append(v)
    hits = {}
    for q in queries[:5]:
        res = hf_get("/api/models?" + urllib.parse.urlencode({"search": q, "sort": "downloads", "direction": -1,
                                                              "limit": 50})) or []
        for x in res:
            owner, name = key(x["id"].split("/")[0]), key(x["id"].split("/")[-1])
            if (any(name.startswith(k) and HF_SUFFIX.match(name[len(k):]) for k in keys)
                    and any(owner.startswith(o) or o.startswith(owner) for o in owners if len(o) > 2)):
                hits[x["id"]] = x.get("downloads") or 0
    return max(hits, key=hits.get) if hits else None


def text_config(cfg):
    for k in TEXT_CONFIGS:
        if isinstance(cfg.get(k), dict):
            return cfg[k]
    return cfg


def hf_record(repo):
    """Exact parameter count (model info API, also for gated repos) and, from a public config.json, the
    architecture, number of experts and native context length."""
    info = hf_get(f"/api/models/{repo}?expand%5B%5D=safetensors&expand%5B%5D=gated") or {}
    rec = {"repo": repo, "gated": info.get("gated"), "parameters": (info.get("safetensors") or {}).get("total")}
    cfg = hf_get(f"/{repo}/resolve/main/config.json") if not info.get("gated") else None
    if cfg:
        lm = text_config(cfg)
        experts = next((lm.get(k) or cfg.get(k) for k in EXPERT_KEYS if (lm.get(k) or cfg.get(k))), None)
        model_type = str(lm.get("model_type") or cfg.get("model_type") or "")
        rec["n_experts"] = experts if isinstance(experts, int) and experts > 1 else None
        rec["architecture"] = ("moe" if rec["n_experts"] else "ssm" if "mamba" in model_type
                               else "dense" if lm.get("num_hidden_layers") else None)
        rec["context_window"] = lm.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    return rec


def fill_hf(df, offline, refresh=True):
    """Fill from the Hugging Face Hub, anonymously, for models that are open-weight or of unknown openness. With
    refresh=False, models already in the cache are not queried again."""
    providers = sorted(set(df.provider.dropna()), key=len, reverse=True)
    cache = json.loads(HF_CACHE.read_text(encoding="utf-8")) if HF_CACHE.exists() else {}
    todo = df[(df.kind == "model") & df.open_weights.ne(False)]
    # Hugging Face owners Epoch gives per provider (e.g. Zhipu → zai-org), for models Epoch has no id for.
    provider_owners = {p: {key(x) for x in g.dropna()} for p, g in df.groupby("provider").hf_developer}
    if not offline:
        for r in todo.itertuples():
            if not refresh and r.normalized_name in cache:
                continue
            try:
                repo = hf_repo(r, providers, provider_owners)
                cache[r.normalized_name] = hf_record(repo) if repo else None
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  HF Hub: {r.normalized_name}: {e}; keeping the cached record")
        HF_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    recs = [cache.get(n) if n in set(todo.normalized_name) else None for n in df.normalized_name]
    g = lambda f: [rec.get(f) if rec else None for rec in recs]
    df["hf_repo"] = g("repo")
    counts = {
        # A repo found under the model's own name and owner means its weights are public.
        "open_weights": fill(df, "open_weights", [True if rec else None for rec in recs], "hf hub"),
        "parameters": fill(df, "parameters", [float(v) if v else None for v in g("parameters")], "hf hub"),
        "architecture": fill(df, "architecture", g("architecture"), "hf config"),
        "n_experts": fill(df, "n_experts", g("n_experts"), "hf config"),
        "context_window": fill(df, "context_window", g("context_window"), "hf config"),
    }
    counts["moe"] = fill(df, "moe", df.architecture.map({"moe": True, "dense": False}), "hf config")
    df["n_experts"] = pd.to_numeric(df.n_experts).astype("Int64")
    df["context_window"] = pd.to_numeric(df.context_window).astype("Int64")
    df["parameters"] = pd.to_numeric(df.parameters)
    print(f"  HF Hub: {sum(r is not None for r in recs)} of {len(todo)} open/unknown models found "
          f"({sum(bool(r and r.get('gated')) for r in recs)} gated: no config without a token) · filled {counts}")
    return df


def model_table(models, offline=False, hf_refresh=True):
    """The model attributes table for `models` (one row per normalized_name, the columns of benchmark_models()):
    Epoch, then the names and runs, models.dev, ReleasedModels and the Hugging Face Hub. Also used by
    assessor_data.py, which collects the models from the converter's JSONL instead of data/processed/.
    hf_refresh=False queries the Hugging Face Hub only for models not in its cache."""
    if not offline:
        download()
    epoch = clean()
    models = fill_hf(fill_external(fill_capabilities(fill_sizes(match(models, epoch))), offline), offline,
                     hf_refresh)
    order = ["normalized_name", "kind", "benchmarks", "display_names", "n_subjects", "provider",
             "release_date", "release_date_source", "epoch_model", "models_dev_id", "releasedmodels_slug", "hf_repo", "n_candidates",
             "open_weights", "open_weights_source", *FIELDS.values(), "parameters_source",
             "active_parameters", "active_parameters_source", "architecture", "architecture_source", "moe", "moe_source",
             "n_experts", "n_experts_source", "multimodal_input", "multimodal_input_source", "context_window", "context_window_source",
             "knowledge_cutoff", "knowledge_cutoff_source", "thinking", "thinking_source"]
    m = models[models.kind == "model"]
    group = m.open_weights.map({True: "open", False: "closed"}).fillna("unknown")
    print(f"matched {m.epoch_model.notna().sum()} of {len(m)} models: {m.match_stage.value_counts().to_dict()} · "
          f"{group.value_counts().to_dict()}")
    # thinking here is whether the model has a thinking mode; the benchmark files' `thinking` is whether a run
    # thought. Renamed so the two do not collide when joining on normalized_name.
    return models[order].rename(columns={"thinking": "thinking_mode", "thinking_source": "thinking_mode_source"})


def main(offline=False):
    table = model_table(benchmark_models(), offline)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(MODELS, index=False)
    print(f"→ {MODELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv[1:])
