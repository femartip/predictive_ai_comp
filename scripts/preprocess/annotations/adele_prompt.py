"""ADeLe prompts for the distilled annotator: rubric loading, the training prompt format, token ids per (item, rubric).

Shared by bank_annotation.py (vLLM, cremaet) and bank_annotation_hf.py (transformers, VRAIN cluster), so both engines
score exactly the same token ids. The format is the one used for the six measurement-db benchmarks
(NeurIPS-assessor-competition/cluster/adele/annotate_adele.py), itself a verbatim copy of delean-batch-manager's
training format:

    system: ""
    user:   The following rubric describes six distinct levels of *<name>* ...<rubric>... TASK INSTANCE: <text>
            INSTRUCTION: ... CHAIN-OF-THOUGHTS REASONING STEPS to score the level of *<name>* ...
    assistant (start of the trained answer): "Based on the rubric's level descriptions, the task instance's demands
            most closely match the criteria described for level "

The next token is the level: the probabilities of "0"–"5" at that position are the annotation.
"""

from pathlib import Path

ANSWER_PREFIX = ("Based on the rubric's level descriptions, the task instance's demands most "
                 "closely match the criteria described for level ")
RUBRICS_V1 = Path(__file__).resolve().parent / "rubrics" / "adele_v1"


def load_rubrics(folders, codes=None):
    """{code: (full_name, content)} from '# Full Name' + body files (delean-batch-manager format)."""
    files = {}
    for folder in [folders] if isinstance(folders, (str, Path)) else folders:
        found = sorted(Path(folder).glob("*.txt"))
        if not found:
            raise FileNotFoundError(f"no rubric .txt files in {folder}")
        for f in found:
            if f.stem in files:
                raise ValueError(f"rubric code {f.stem} in both {files[f.stem].parent} and {folder}")
            files[f.stem] = f
    if codes:
        unknown = set(codes) - set(files)
        if unknown:
            raise ValueError(f"unknown rubric codes: {sorted(unknown)}")
        files = {c: files[c] for c in codes}
    out = {}
    for code, path in files.items():
        # the training files had LF line endings: never let a carriage return into a prompt
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        lines = text.splitlines(keepends=True)
        out[code] = (lines[0].strip().lstrip("#").strip(), "".join(lines[1:]).strip())
    return out


def full_instruction(subdomain, rubric_content, prompt):
    """Verbatim copy of delean-batch-manager's get_full_instruction (the training format)."""
    instruction = (
        'INSTRUCTION: '
        f'Score the level of *{subdomain}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. '
        'Use CHAIN-OF-THOUGHTS REASONING to reason step by step before assigning the score. '
        'After the CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the statement: '
        f'"Thus, the level of *{subdomain}* demanded by the given TASK INSTANCE is: SCORE"'
        ', where SCORE is an integer score you have determined.')
    return (
        f"The following rubric describes six distinct levels of *{subdomain}* required by different tasks:"
        f"\n{rubric_content}\n\nTASK INSTANCE: {prompt}"
        f"\n\n{instruction}"
        f'\n\nCHAIN-OF-THOUGHTS REASONING STEPS to score the level of *{subdomain}* demanded by the given TASK INSTANCE above:'
    )


def skeletons(tok, rubrics):
    """{code: (head, tail, head_ids, tail_ids)}: the chat-formatted prompt before and after the item text."""
    out = {}
    for code, (name, content) in rubrics.items():
        s = tok.apply_chat_template([{"role": "system", "content": ""},
                                     {"role": "user", "content": full_instruction(name, content, "\x00")}],
                                    tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX
        head, tail = s.split("\x00")
        out[code] = (head, tail, tok(head, add_special_tokens=False)["input_ids"],
                     tok(tail, add_special_tokens=False)["input_ids"])
    return out


def prompt_ids(tok, items, sk, max_len, tail_frac=0.75):
    """Token ids per (item, rubric) for a chunk of items ({text_id, content}).

    The whole prompt is tokenised at once (as in training), in one batched call per rubric. An item whose prompt is
    longer than max_len keeps its first and last tokens (tail_frac of the item budget at the end, where tasks such as
    ResearchCodeBench put the instructions), joined by "[...]".
    """
    marker = tok("\n[...]\n", add_special_tokens=False)["input_ids"]
    rows = []
    for code, (head, tail, head_ids, tail_ids) in sk.items():
        enc = tok([head + it["content"] + tail for it in items], add_special_tokens=False)["input_ids"]
        budget = max_len - len(head_ids) - len(tail_ids) - len(marker)
        for it, ids in zip(items, enc):
            truncated = len(ids) > max_len
            if truncated:
                item_ids = tok(it["content"], add_special_tokens=False)["input_ids"]
                n_tail = int(budget * tail_frac)
                ids = head_ids + item_ids[:budget - n_tail] + marker + item_ids[len(item_ids) - n_tail:] + tail_ids
            rows.append({"text_id": it["text_id"], "rubric": code, "ids": ids, "truncated": truncated})
    return rows


def done_pairs(out_dir):
    """(text_id, rubric) pairs already written in out_dir's CSV parts (any depth): resuming skips them."""
    import pandas as pd
    done = set()
    for f in Path(out_dir).rglob("part-*.csv"):
        d = pd.read_csv(f, usecols=["text_id", "rubric"])
        done |= set(zip(d["text_id"], d["rubric"]))
    return done
