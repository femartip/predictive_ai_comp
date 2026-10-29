# ADeLe v1.0 demand levels of the AI Measurement Data Bank, slim copy

One Parquet file per benchmark, one row per annotated item, in one subfolder per conversion run of the bank:
no-assets/ (benchmarks without assets), asset-image/ and asset-audio/ (the annotator saw each image or audio clip
through its written description), asset-text/ and asset-application/ (files inlined in the item text: CSV, SQLite
tables, NDJSON, PDF text..., cut by the converter at about 24,400 tokens). Columns: item_id, text_id (items with identical text share it),
truncated_any (the item was cut to the annotator's 32,767-token window), digit_mass_min (probability mass the
annotator put on the six level tokens; near 1 = it answered with a level) and, for each of the 18 rubrics (AS AT CEc
CEe CL KNa KNc KNf KNn KNs MCr MCt MCu MS QLl QLq SNs VO), <code>_exp: the expected level (sum of k * p_k, 0-5).

Produced by scripts/preprocess/annotations/ (distilled annotator ceferra/qwen2.5-7b-adele-annotator) on at most 600
items per benchmark, chosen deterministically (select_items.py); copied here by scripts/assessors_v3/export_adele.py
from the full tables on vent (/data/datasets/annotations/adele_v1, which also keep level probabilities). Read by
scripts/assessors_v3/build_data.py. No item text is stored.
