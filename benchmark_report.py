"""Scores of the official benchmark, split into a development and a held-out part.

The official benchmark (`benchmarking/evaluate_results.py` in SPOT-data-generation) writes one
row per gold sentence with all its counts. This script adds them up again for two fixed parts
of the gold set, so a prompt change can be developed on one part and checked on the other:

    python benchmark_report.py run_eval.xlsx                      # one run
    python benchmark_report.py base_*_eval.xlsx --vs new_*_eval.xlsx   # mean of runs, and the change
    python benchmark_report.py run_eval.xlsx --failures dev       # the dev sentences that failed

The part a sentence belongs to depends only on its text (`is_holdout`), so it stays the same
when the gold sheet is reordered or extended. Read the failures of the dev part only; the
held-out part is for the scores.
"""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd

HOLDOUT_SHARE = 3     # every third sentence, by hash

# (label, column with the correct count(s), column with the total), as in evaluate_results.py
RATIOS = [
    ("area", ("num_correct_name_area", "num_correct_bbox"), "total_area"),
    ("entity perfect match", ("num_entity_match_perfect",), "total_ref_entities"),
    ("entity type", ("num_correct_entity_type",), "total_ref_entities"),
    ("cluster distance", ("num_correct_cluster_distance",), "total_clusters"),
    ("cluster points", ("num_correct_cluster_points",), "total_clusters"),
    ("properties", ("num_correct_properties_perfect",), "total_properties"),
    ("cuisine property", ("num_correct_cuisine_properties",), "total_cuisine_property"),
    ("colour property", ("num_correct_color",), "total_color_property"),
    ("distance relations", ("num_correct_dist_rels",), "total_dist_rels"),
    ("contains relations", ("num_correct_contains_rels",), "total_contains_rels"),
    ("relative spatial terms", ("num_correct_relative_spatial_terms",), "total_relative_spatial_terms"),
]
COUNTS = ["num_missing_entity", "num_hallucinated_entity", "num_missing_properties",
          "num_hallucinated_properties"]
META = ["1 entity", "2 entities", "3 entities", "distance relation", "area", "proporties",
        "typos", "grammar mistakes", "rel spatial term", "cluster", "contains relation",
        "brand/name as property", "brand/name standalone", "non-roman alphabet"]
PARTS = ("dev", "holdout", "all")


def is_holdout(sentence: str) -> bool:
    """Whether a gold sentence belongs to the held-out part (about a third of them)."""
    digest = hashlib.sha1(" ".join(str(sentence).lower().split()).encode("utf-8")).hexdigest()
    return int(digest, 16) % HOLDOUT_SHARE == 0


def load(path: str) -> pd.DataFrame:
    rows = pd.read_excel(path, index_col=0)
    rows["holdout"] = rows["sentence"].map(is_holdout)
    return rows


def part_of(rows: pd.DataFrame, part: str) -> pd.DataFrame:
    return rows if part == "all" else rows[rows["holdout"] == (part == "holdout")]


def scores(rows: pd.DataFrame) -> dict[str, float]:
    """The benchmark's scores for these rows; ratios in percent, counts as they are."""
    result = {"sentences": len(rows), "perfect match": 100 * rows["is_perfect_match"].mean()}
    for label, correct, total in RATIOS:
        if rows[total].sum():
            result[label] = 100 * sum(rows[c].sum() for c in correct) / rows[total].sum()
    for column in COUNTS:
        result[column.replace("num_", "").replace("_", " ")] = float(rows[column].sum())
    for field in META:
        tagged = rows[rows[field] == 1] if field in rows else rows.iloc[:0]
        if len(tagged):
            result[f"perfect match: {field}"] = 100 * tagged["is_perfect_match"].mean()
    return result


def mean_scores(paths: list[str]) -> dict[str, pd.Series]:
    """Per part, the mean over the runs and the lowest and highest perfect match."""
    runs = [load(path) for path in paths]
    result = {}
    for part in PARTS:
        table = pd.DataFrame([scores(part_of(rows, part)) for rows in runs])
        result[part] = table.mean()
        result[part + " range"] = (table["perfect match"].min(), table["perfect match"].max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="per-sentence result files of the official benchmark (_eval.xlsx)")
    parser.add_argument("--vs", nargs="+", help="result files of a second variant; shows the change")
    parser.add_argument("--failures", choices=PARTS, help="print the sentences of this part without a perfect match")
    args = parser.parse_args()

    if args.failures:
        rows = part_of(load(args.runs[0]), args.failures)
        failed = rows[~rows["is_perfect_match"]]
        for _, row in failed.iterrows():
            print(f"### {row['sentence']}\n--- gold\n{str(row['yaml_true_string']).strip()}\n"
                  f"--- prediction\n{str(row['yaml_pred_string']).strip()}\n")
        print(f"{len(failed)} of {len(rows)} sentences in '{args.failures}' failed.")
        return

    base = mean_scores(args.runs)
    other = mean_scores(args.vs) if args.vs else None
    print(f"{len(args.runs)} run(s)" + (f" vs {len(args.vs)} run(s); change in brackets" if other else ""))
    print(f"{'':38}" + "".join(f"{part:>22}" for part in PARTS))
    for key in base["all"].index:
        cells = []
        for part in PARTS:
            value = base[part].get(key)
            if value is None or pd.isna(value):
                cells.append(f"{'':>22}")
                continue
            if other is None:
                cells.append(f"{value:>22.1f}")
            else:
                new = other[part].get(key, float("nan"))
                cells.append(f"{new:>13.1f} ({new - value:+5.1f})")
        print(f"{key:<38}" + "".join(cells))
    for name, table in (("", base), ("--vs ", other)):
        if table and len(args.vs if name else args.runs) > 1:
            print(f"{name}perfect match, lowest to highest run: " + ", ".join(
                f"{part} {table[part + ' range'][0]:.1f}-{table[part + ' range'][1]:.1f}" for part in PARTS))


if __name__ == "__main__":
    main()
