"""
Decisions:
- Items are the build's items: a final-answer problem under one prompt wording, or one grading criterion of a
  proof problem (points / max points). Several build items share a raw_item_id (<competition>::<problem_idx>):
  the criteria of a proof problem, and problems whose prompt differs between model configurations. item_id =
  matharena:<raw_item_id> when the raw_item_id is unique, else matharena:<raw_item_id>::<build item_id>.
  Problems reused across competitions (e.g. aime_2025 and aime_2025_I) stay separate items, as in the build.
- question = the user message the model received. Image problems append one marker <image:PATH> per image,
  PATH relative to data/matharena/ (the images are written there from assets.parquet).
- ground_truth = the gold answer for final-answer problems; for proof criteria, the criterion (title, grading
  scheme, max points) that the judge applied.
- Drop responses whose released solution is empty ("", "None") and that are graded 0: the API returned
  nothing, so the 0 says nothing about the problem (138 attempts, most from one batch of Claude Opus 4.7 runs).
  The two empty-solution attempts graded 1 are kept: the grade shows an answer existed, only its text is
  missing. Proof criteria after the first carry no trace by design (the build attaches the solution to the
  first criterion only) and are kept.
- Subjects: half of the build's subjects have a blank normalized_name. NAMES fills it with the model's
  canonical name (the build's convention: "<Provider> <Model>"; variants such as reasoning effort, dated
  snapshots or agent setups map to the base model, as the build does for its filled names). Provider and
  release_date then come from models.yaml where the build leaves them blank; reasoning_effort from the
  "(low|medium|high|xhigh|max)" suffix of the display name (the build's own effort_pattern).
- thinking is set per variant: True when the name marks reasoning (Think, Thinking, Reasoning, an effort
  level, R1 and its distills, o-series), otherwise the model's default from models.yaml.
- MathArena's agent setups (display names with "Agent"/"agent") are multi-turn and agentic; every other run
  is a single turn.
- score = mean over the released attempts (trials); n_runs = how many.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from common import DATA, benchmark_columns, huggingface_provenance, load_curated, load_metadata, subject_columns, write

B = "matharena"
REPO = "aims-foundations/measurement-db"
IMAGE_MARKER = "<image:{}>"  # parse with re.findall(r"<image:([^>]+)>", question)
EVAL_TYPE = {"exact_matcher": "exact_match", "judge": "judge"}
EMPTY_SOLUTION = {"", "None", "null"}
EFFORT = re.compile(r"\((low|medium|high|xhigh|max)\)", re.I)
THINKING_MARK = re.compile(r"think|reasoning|\bR1\b|distill|\bo[134]\b|o[134]-|\((?:low|medium|high|xhigh|max)\)", re.I)
AGENT_MARK = re.compile(r"\bagent\b", re.I)

# display_name → normalized_name for the subjects the build leaves blank.
NAMES = {
    "Claude-3.7-Sonnet (Think)": "Anthropic Claude 3.7 Sonnet",
    "Claude-Opus-4.0 (Think)": "Anthropic Claude Opus 4",
    "Claude-Sonnet-4.5 (Think)": "Anthropic Claude Sonnet 4.5",
    "DeepSeek-R1-Distill-1.5B": "DeepSeek R1 Distill Qwen 1.5B",
    "DeepSeek-R1-Distill-14B": "DeepSeek R1 Distill Qwen 14B",
    "DeepSeek-R1-Distill-32B": "DeepSeek R1 Distill Qwen 32B",
    "DeepSeek-R1-Distill-70B": "DeepSeek R1 Distill Llama 70B",
    "DeepSeek-V3-03-24": "DeepSeek V3",
    "DeepSeek-v3.1 (Think)": "DeepSeek V3.1",
    "DeepSeek-v3.2-Exp (Think)": "DeepSeek V3.2",
    "DeepSeek-v3.2-Speciale": "DeepSeek V3.2 Speciale",
    "DeepSeek-v3.2-Speciale Agent": "DeepSeek V3.2 Speciale",
    "GLM 4.5": "Zhipu GLM-4.5",
    "GLM 4.5 Air": "Zhipu GLM-4.5 Air",
    "GLM 4.5V": "Zhipu GLM-4.5V",
    "GLM 4.6": "Zhipu GLM-4.6",
    "GLM 4.7": "Zhipu GLM-4.7",
    "GPT OSS 120B (high)": "OpenAI gpt-oss-120b",
    "GPT OSS 20B (high)": "OpenAI gpt-oss-20b",
    "GPT-5 (High) Agent": "OpenAI GPT-5",
    "GPT-5 (high)": "OpenAI GPT-5",
    "GPT-5-mini (high)": "OpenAI GPT-5 mini",
    "GPT-5-nano (high)": "OpenAI GPT-5 nano",
    "GPT-5-nano (low)": "OpenAI GPT-5 nano",
    "GPT-5.1 (high)": "OpenAI GPT-5.1",
    "Gemini 2.5 Flash (Thinking)": "Google Gemini 2.5 Flash",
    "gemini-2.5-flash (think)": "Google Gemini 2.5 Flash",
    "Gemini 2.5 Pro (05-06)": "Google Gemini 2.5 Pro",
    "Gemini 2.5 Pro (agent)": "Google Gemini 2.5 Pro",
    "Gemini 2.5 Pro (best-of-32)": "Google Gemini 2.5 Pro",
    "gemini-2.5-pro*": "Google Gemini 2.5 Pro",
    "Gemini Deep Think 3 (12/25)": "Google Gemini 3 Deep Think",
    "Gemini IMO Deep Think": "Google Gemini 2.5 Deep Think",
    "gemini-2.0-flash-thinking": "Google Gemini 2.0 Flash Thinking",
    "Grok 3 Mini (high)": "xAI Grok 3 Mini",
    "Grok 3 Mini (low)": "xAI Grok 3 Mini",
    "Grok 4 (Specific Prompt)": "xAI Grok 4",
    "Grok 4 Fast R": "xAI Grok 4 Fast",
    "Grok 4 Fast (Reasoning) (Selfcheck Agent)": "xAI Grok 4 Fast",
    "K2-Think": "MBZUAI K2-Think",
    "LIMO": "GAIR LIMO",
    "OpenThinker-32B": "Open Thoughts OpenThinker 32B",
    "Qwen3-235B-2507-Think": "Alibaba Qwen3 235B A22B",
    "Qwen3-VL-235B Instruct": "Alibaba Qwen3-VL 235B A22B",
    "o1 (medium)": "OpenAI o1",
    "o1-pro (high)": "OpenAI o1-pro",
    "o3-mini (high)": "OpenAI o3-mini",
    "o3-mini (low)": "OpenAI o3-mini",
    "o3-mini (medium)": "OpenAI o3-mini",
    "o4-mini (high)": "OpenAI o4-mini",
    "o4-mini (low)": "OpenAI o4-mini",
    "o4-mini (medium)": "OpenAI o4-mini",
}


def cached_source_files(provenance):
    """Fetch the required files into the Hub cache at the run's pinned revision."""
    filenames = sorted(set(provenance["source_files"]))

    def download(filename):
        return Path(hf_hub_download(
            repo_id=REPO, filename=filename, repo_type="dataset",
            revision=provenance["source_revision"],
        ))

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(filenames, pool.map(download, filenames)))


def fill_subjects(subjects):
    """Fill normalized_name, provider, release_date and reasoning_effort where the build leaves them blank."""
    curated = load_curated()
    s = subjects.copy()
    blank = s.normalized_name.isna()
    missing = sorted(set(s.display_name[blank]) - set(NAMES))
    assert not missing, f"NAMES has no entry for {missing}"
    s.loc[blank, "normalized_name"] = s.display_name[blank].map(NAMES)
    for field in ("provider", "release_date"):
        fallback = s.normalized_name.map(lambda n: (curated.get(n) or {}).get(field))
        s[field] = s[field].fillna(fallback)
    effort = s.display_name.str.extract(EFFORT, expand=False).str.lower()
    s["reasoning_effort"] = s.reasoning_effort.fillna(effort)
    return s


def thinking(subjects):
    """Per-variant thinking: name markers first, else the model's default in models.yaml (None if switchable)."""
    curated = load_curated()
    value = subjects.normalized_name.map(lambda n: (curated.get(n) or {}).get("thinking")).astype(object)
    value[subjects.display_name.str.contains(THINKING_MARK)] = True
    value[subjects.display_name.str.contains("Instruct")] = False
    return value.where(value.notna(), None)


def write_images(items, assets):
    """Write each item's images under data/matharena/images; return item_id → markers to append."""
    out = DATA / B / "images"
    out.mkdir(parents=True, exist_ok=True)
    blobs = assets.set_index("asset_id").data
    markers = {}
    for item_id, manifest in zip(items.item_id, items.asset_manifest):
        if not isinstance(manifest, str):
            continue
        refs = []
        for a in sorted(json.loads(manifest), key=lambda a: a["ordinal"]):
            ext = a["media_type"].split("/")[-1]
            rel = f"images/{a['asset_id']}.{ext}"
            path = DATA / B / rel
            if not path.exists():
                path.write_bytes(blobs[a["asset_id"]])
            refs.append(IMAGE_MARKER.format(rel))
        markers[item_id] = "\n" + "\n".join(refs)
    return markers


def ground_truth(criterion):
    c = json.loads(criterion)
    if c.get("reference_answer") is not None:
        return c["reference_answer"]
    return c.get("rule")  # proof criterion: title, grading scheme and max points, as the judge received them


def main():
    tables = {name: f"{B}/formatted_tables/{name}.parquet"
              for name in ("items", "subjects", "responses", "traces", "assets")}
    provenance = huggingface_provenance(REPO, tables.values())
    provenance["source_files"] = sorted([*tables.values(), f"{B}/metadata.yaml"])
    print(f"Loading {B} from Hugging Face revision {provenance['source_revision']}", flush=True)
    files = cached_source_files(provenance)
    meta = load_metadata(B, REPO, provenance["source_revision"])

    items = pd.read_parquet(files[tables["items"]])
    resp = pd.read_parquet(files[tables["responses"]])
    traces = pd.read_parquet(files[tables["traces"]], columns=["response_id", "trace"])

    # Responses: drop API failures (released solution is empty and graded 0).
    solution = resp.response_id.map(traces.set_index("response_id").trace)
    api_failure = solution.notna() & solution.str.strip().isin(EMPTY_SOLUTION) & resp.response.eq(0)
    kept = resp[~api_failure]
    scores = kept.groupby(["subject_id", "item_id"]).response.agg(score="mean", n_runs="size").reset_index()

    # Items: one per final-answer problem (per prompt wording) or proof criterion.
    verifier = items.verifier.map(lambda v: json.loads(v)["class"])
    shared = items.raw_item_id.duplicated(keep=False)
    markers = write_images(items, pd.read_parquet(files[tables["assets"]]))
    items = pd.DataFrame({
        "build_item_id": items.item_id,
        "item_id": f"{B}:" + items.raw_item_id + ("::" + items.item_id).where(shared, ""),
        "question": items.content + items.item_id.map(markers).fillna(""),
        "ground_truth": items.grading_criterion.map(ground_truth),
        "eval_type": verifier.map(EVAL_TYPE),
        "modality": items.asset_manifest.notna().map({True: "image;text", False: "text"}),
    })
    assert items.item_id.is_unique

    subjects = fill_subjects(pd.read_parquet(files[tables["subjects"]]))
    subj = subject_columns(subjects)
    variant = subjects.set_index("subject_id")
    subj["thinking"] = subj.subject_id.map(thinking(subjects).set_axis(subjects.subject_id))
    subj["reasoning_effort"] = subj.reasoning_effort.where(subj.reasoning_effort.notna() | ~subj.thinking.eq(False),
                                                           "none")
    agent = subj.subject_id.map(variant.display_name.str.contains(AGENT_MARK))

    bench = benchmark_columns(meta)
    for column in ("modality", "multi_turn", "agentic"):
        bench.pop(column)  # set per item / per subject
    df = (scores.rename(columns={"item_id": "build_item_id"}).merge(items, on="build_item_id")
          .merge(subj.assign(multi_turn=agent, agentic=agent), on="subject_id", how="left")
          .assign(benchmark=B, programming_language=None, tools=None, **bench))
    assert len(df) == len(scores), "lost rows when joining items or subjects"

    path = write(df, B, provenance)
    print(f"{len(df):,} rows ({df.subject_id.nunique()} subjects × {df.item_id.nunique():,} items) from "
          f"{len(kept):,} attempts → {path.relative_to(DATA.parent)}\n"
          f"dropped {api_failure.sum()} attempts with an empty solution (API failures): "
          f"{resp[api_failure].subject_id.map(variant.display_name).value_counts().to_dict()}\n"
          f"filled normalized_name for {len(NAMES)} display names; {len(markers)} items with images")


if __name__ == "__main__":
    main()
