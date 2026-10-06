"""
Decisions:
- An item is a question under one test condition: item_id = mmdocrag:<q_id>_<mode>_<n_quotes>, e.g.
  mmdocrag:1_pure-text_15. The 15- and 20-quote sets are sampled and ordered differently, and the mode changes
  the input (image descriptions vs images), so each condition is its own item.
- question = exactly what the model received, rebuilt as upstream inference_wrapper.py does: system prompt,
  text quotes, image quotes and the question, serialised as "[system]\n...\n\n[user]\n...". In multimodal
  items each image is the marker <image:PATH>, PATH relative to data/mmdocrag/ (images.zip from the
  MMDocIR/MMDocRAG Hugging Face dataset).
- ground_truth = the short answer and the perfect (interleaved) answer, both of which the judge receives.
- Drop responses without a score, responses whose raw output is an API error (error field or empty response)
  and the five Qwen2.5 LoRA models fine-tuned on this task by the benchmark authors (display_name *-ft).
- The pure-text OCR runs in raw/resp are not part of the build and are not used.
- thinking is set per released variant: the build's normalized_name merges e.g. qwen3-8b and qwen3-8b-no-think.
  Open-weight models are checked against <think> blocks in their raw outputs. Closed API models hide their
  reasoning, so their values are assumed from the model name and the API's documented default (see THINKING).
- release_date is the build's, unchanged, except that a month-only date (qwen-plus: 2023-09) is padded to the
  first of the month (2023-09-01) so every subject has YYYY-MM-DD.
- modality is per item: text for pure-text, image;text for multimodal.
- score = the judge's grade (mean of five 0-5 dimensions / 5); one run per (subject, item).
- The output repeats each ~10k-character question once per subject (~3.5 GB), so text columns stay
  categorical through the joins and the file is written in chunks; one in-memory frame would exhaust RAM.
"""

import json
import re
import tarfile

import pandas as pd
from common import COLUMNS, DATA, OUT, benchmark_columns, load_metadata, subject_columns

B = "mmdocrag"
DIR = DATA / B
RAW = DIR / "raw"
IMAGE_MARKER = "<image:{}>"  # parse with re.findall(r"<image:([^>]+)>", question)
SYSTEM_PROMPT = {"pure-text": "pure_text_infer.txt", "multimodal": "multimodal_infer.txt"}
MODALITY = {"pure-text": "text", "multimodal": "image;text"}
RESP_FILE = re.compile(r"^(?P<model>.+)_(?P<mode>pure-text|multimodal)(?:_response)?_quotes(?P<n>\d+)"
                       r"(?:_response)?\.jsonl$")

# Whether each released variant reasons before answering.
# - Open-weight models: verified against raw/resp (thinking variants emit <think> in ~100% of outputs, the
#   others in 0%).
# - Closed API models (Claude, Gemini, GPT/o3, Grok, Qwen API models): NOT verifiable from the data, since the APIs
#   hide reasoning; assumed from the name and the API default. Notably gemini-2.5-flash/-pro, gemini-2.0-flash-tk,
#   gpt-o3-mini and grok-3-mini-beta = True; the authors could have disabled thinking without it showing here.
THINKING = {
    "claude-3.5-sonnet": False, "deepseek-r1": True, "deepseek-r1-distill-llama-70b": True,
    "deepseek-r1-distill-qwen-32b": True, "deepseek-v3": False, "gemini-1.5-pro": False,
    "gemini-2.0-flash": False, "gemini-2.0-flash-tk": True, "gemini-2.0-pro": False, "gemini-2.5-flash": True,
    "gemini-2.5-pro": True, "gpt-4-turbo": False, "gpt-4.1": False, "gpt-4.1-mini": False, "gpt-4.1-nano": False,
    "gpt-4o": False, "gpt-4o-mini": False, "gpt-o3-mini": True, "grok-3-beta": False, "grok-3-mini-beta": True,
    "internvl2.5-8b": False, "internvl2.5-26b": False, "internvl2.5-38b": False, "internvl2.5-78b": False,
    "internvl3-8B": False, "internvl3-9B": False, "internvl3-14B": False, "internvl3-38b": False,
    "internvl3-78b": False, "janus-pro-7b": False, "llama3.1-8b": False, "llama3.2-3b": False,
    "llama3.3-70b": False, "llama4-mave-17b-128e": False, "llama4-scout-17b-16e": False,
    "minicpm-o-2.6-8b": False, "mistral-7b": False, "mistral-small-24b": False, "mixtral-8x7b": False,
    "qvq-max-no-think": False, "qwen-qvq-max": True, "qwen-max": False, "qwen-plus": False,
    "qwen-qwq-plus": True, "qwen-vl-max": False, "qwen-vl-plus": False, "qwen2.5-3b": False,
    "qwen2.5-7b": False, "qwen2.5-14b": False, "qwen2.5-32b": False, "qwen2.5-72b": False,
    "qwen2.5-vl-7b": False, "qwen2.5-vl-32b": False, "qwen2.5-vl-72b": False, "qwen3-4b-no-think": False,
    "qwen3-8b": True, "qwen3-8b-no-think": False, "qwen3-14b": True, "qwen3-14b-no-think": False,
    "qwen3-30b-a3b": True, "qwen3-30b-a3b-no-think": False, "qwen3-32b": True, "qwen3-235b-a22b": True,
}


def system_prompts():
    """Inference system prompts from the pinned upstream source archive, read as upstream does (unstripped)."""
    archive = next((RAW / "reproducibility").glob("source-*.tar.gz"))
    with tarfile.open(archive) as tar:
        members = {m.name.rsplit("/", 1)[-1]: m for m in tar.getmembers() if "/prompt_bank/" in m.name}
        return {mode: tar.extractfile(members[name]).read().decode("utf-8") for mode, name in SYSTEM_PROMPT.items()}


def user_message(q, mode):
    """User turn of upstream get_text_messages (pure-text) / get_interleaved_messages (multimodal)."""
    texts = "Text Quotes are:" + "".join(f"\n[{i}] {t['text']}" for i, t in enumerate(q["text_quotes"], 1))
    if mode == "pure-text":
        images = "\nImage Quotes are:" + "".join(f"\nimage{i} is described as: {m['img_description']}"
                                                 for i, m in enumerate(q["img_quotes"], 1))
        return f"{texts}{images}\n\nThe user question is: {q['question']}"
    # Multimodal content parts are sent back to back; each image part becomes a path marker.
    images = "Image Quotes are:\n" + "".join(f"- image{i} is " + IMAGE_MARKER.format(m["img_path"])
                                             for i, m in enumerate(q["img_quotes"], 1))
    return f"{texts}\n{images}The user question is: {q['question']}"


def build_items():
    """One row per (q_id, mode, n_quotes) with the exact model input and the judge's references."""
    system = system_prompts()
    rows = []
    for n in (15, 20):
        for line in open(RAW / f"evaluation_{n}.jsonl", encoding="utf-8"):
            q = json.loads(line)
            for mode in SYSTEM_PROMPT:
                rows.append({
                    "q_id": q["q_id"], "test_condition": f"{mode}/quotes{n}",
                    "item_id": f"{B}:{q['q_id']}_{mode}_{n}",
                    "question": f"[system]\n{system[mode]}\n\n[user]\n{user_message(q, mode)}",
                    "ground_truth": f"Short answer: {q['answer_short']}\n\nPerfect answer: {q['answer_interleaved']}",
                    "modality": MODALITY[mode],
                })
    items = pd.DataFrame(rows)
    text = ["item_id", "question", "ground_truth", "modality"]
    return items.astype({c: "category" for c in text})


def api_errors():
    """(model, test_condition, q_id) of raw outputs that are API errors: an error field or an empty response."""
    rows = []
    for path in (RAW / "resp").glob("*.jsonl"):
        m = RESP_FILE.match(path.name)
        if m is None:  # pure-text OCR runs
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "error" in r or not (r.get("response") or "").strip():
                rows.append({"model": m["model"].lower(), "test_condition": f"{m['mode']}/quotes{m['n']}",
                             "q_id": r["q_id"]})
    return pd.DataFrame(rows).drop_duplicates()


def write_chunked(df, chunk=20_000):
    """common.write's checks, but streamed to disk chunk by chunk instead of serialised as one string."""
    missing = set(COLUMNS) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert df.item_id.str.startswith(f"{B}:").all()
    assert not df.duplicated(["item_id", "subject_id"]).any(), "duplicate (item, subject) rows"
    assert df.score.between(0, 1).all(), "score outside [0, 1]"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{B}.jsonl"
    df = df[COLUMNS]
    # Row by row: pandas.to_json builds each chunk as one string, then copies it again for lines=True.
    with open(path, "w", encoding="utf-8") as fh:
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk].astype(object)
            for row in part.where(part.notna(), None).itertuples(index=False):
                fh.write(json.dumps(dict(zip(COLUMNS, row)), ensure_ascii=False) + "\n")
    return path


def main():
    meta = load_metadata(B)
    subjects = pd.read_parquet(DIR / "subjects.parquet")
    finetuned = subjects.display_name.str.endswith("-ft")
    subjects = subjects[~finetuned]
    assert set(subjects.display_name) == set(THINKING), "THINKING must cover exactly the retained subjects"

    resp = pd.read_parquet(DIR / "response.parquet")
    raw_ids = pd.read_parquet(DIR / "items.parquet", columns=["item_id", "raw_item_id"])
    resp = (resp.merge(raw_ids, on="item_id")
            .assign(q_id=lambda d: d.raw_item_id.str.removeprefix("q_id::").astype(int))
            .merge(subjects[["subject_id", "display_name"]], on="subject_id"))  # inner: drops fine-tuned
    n_total = len(resp)
    n_unscored = resp.response.isna().sum()
    resp = resp.dropna(subset=["response"])

    errors = api_errors().assign(api_error=True)
    resp = (resp.assign(model=resp.display_name.str.lower())
            .merge(errors, on=["model", "test_condition", "q_id"], how="left"))
    n_api = resp.api_error.fillna(False).astype(bool).sum()
    resp = resp[~resp.api_error.fillna(False).astype(bool)]

    items = build_items()
    subj = subject_columns(subjects)
    variant = subjects.set_index("subject_id").display_name
    subj["thinking"] = subj.subject_id.map(variant).map(THINKING)
    subj["reasoning_effort"] = subj.reasoning_effort.where(subj.thinking, "none")
    month_only = subj.release_date.str.fullmatch(r"\d{4}-\d{2}")
    subj.loc[month_only, "release_date"] += "-01"

    bench = benchmark_columns(meta)
    bench.pop("modality")  # set per item
    df = (resp[["subject_id", "q_id", "test_condition", "response"]]
          .merge(items, on=["q_id", "test_condition"])
          .merge(subj, on="subject_id", how="left")
          .assign(benchmark=B, score=lambda d: d.response, n_runs=1, eval_type="judge",
                  programming_language=None, tools=None, **bench))
    assert len(df) == len(resp), "lost rows when joining items or subjects"

    path = write_chunked(df)
    print(f"{len(df):,} rows ({df.subject_id.nunique()} subjects × {df.item_id.nunique():,} items) → "
          f"{path.relative_to(DATA.parent)}\n"
          f"dropped: {finetuned.sum()} fine-tuned subjects; of the remaining {n_total:,} responses, "
          f"{n_unscored} unscored and {n_api} API errors")


if __name__ == "__main__":
    main()
