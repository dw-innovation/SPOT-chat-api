"""Run the behaviour test sets (docs/ROADMAP.md, sections A-D) against the real model.

Every case is played like a chat through the API: the query and the history are carried
from turn to turn, and each turn is checked against what the case expects (declined or
not, the query, reported limits, the reply).

    python run_testsets.py                                  # all sets in testsets/
    python run_testsets.py testsets/safety.yaml --workers 4
    python run_testsets.py --section A,B --group self_harm
    python run_testsets.py --gold data/goldstandard_testing_dataset.xlsx   # + derived dialogues
    python run_testsets.py --gold data/gold.xlsx --list     # show the cases, call no model
    python run_testsets.py --repeat 3                       # same result every time?

Writes one row per turn to --out and prints a summary. Turns with a gold query also go to
--jsonl and --gold-out, a prediction file and a gold sheet for SPOT's benchmarking scripts
(the sheet is named "testsets"). Uses the same LLM settings as the app (.env).
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

from evaluate import benchmark_yaml, load_gold, preflight
from spot_chat.config import load_settings, make_client
from spot_chat.descriptors import get_descriptors
from spot_chat.policy import REFUSAL_CATEGORIES
from spot_chat.query import to_yaml
from testsets.cases import Case, TurnResult, load_file, run_case, set_files
from testsets.derive import derive

# Columns SPOT's benchmark expects in a gold sheet (it reports perfect matches per column).
META_FIELDS = ["1 entity", "2 entities", "3 entities", "distance relation", "area", "proporties",
               "typos", "grammar mistakes", "rel spatial term", "cluster", "contains relation",
               "brand/name as property", "brand/name standalone", "non-roman alphabet"]
_NON_LATIN_RE = re.compile(r"[^\u0000-ɏḀ-ỿ -⁯]")


def meta_fields(gold: dict) -> dict:
    """The gold sheet's tag columns, as far as the query itself tells."""
    entities, relations = gold["entities"], gold.get("relations") or []
    properties = [p for e in entities for p in e.get("properties", [])]
    flags = {
        f"{len(entities)} entit{'y' if len(entities) == 1 else 'ies'}": True,
        "distance relation": any(r["type"] == "distance" for r in relations),
        "contains relation": any(r["type"] == "contains" for r in relations),
        "area": gold["area"]["type"] == "area",
        "proporties": bool(properties),
        "cluster": any(e["type"] == "cluster" for e in entities),
        "brand/name as property": any(p["name"].lower() in ("name", "brand", "brand name") for p in properties),
        "brand/name standalone": any(e["name"].lower().startswith("brand:") for e in entities),
        "non-roman alphabet": bool(_NON_LATIN_RE.search(yaml.safe_dump(gold, allow_unicode=True))),
    }
    return {name: int(flags.get(name, False)) for name in META_FIELDS}


def collect(args) -> tuple[dict[str, str], list[Case]]:
    """Section titles and the cases selected on the command line."""
    titles, cases = {}, []
    for path in args.files or set_files():
        section, title, found = load_file(path)
        titles[section] = title
        cases += found
    if args.gold and args.derive:
        gold = load_gold(args.gold, args.sheet)
        cases += derive(list(zip(gold["sentence"], gold["YAML"])), args.derive, args.seed)
        titles.setdefault("C", "Multi-turn behaviour")
    sections = {s.strip().upper() for s in args.section.split(",")} if args.section else None
    groups = {g.strip() for g in args.group.split(",")} if args.group else None
    cases = [c for c in cases
             if (not sections or c.section in sections) and (not groups or c.group in groups)]
    seen = set()
    for case in cases:
        if case.id in seen:
            raise SystemExit(f"Case id '{case.id}' is used twice.")
        seen.add(case.id)
    return titles, cases[:args.limit] if args.limit else cases


def show(cases: list[Case]) -> None:
    for case in cases:
        print(f"\n[{case.section}] {case.group} / {case.id}")
        for turn in case.history:
            print(f"    ({turn['role']}) {turn['content']}")
        for turn in case.turns:
            print(f"    > {turn.user}")
            expect = turn.expect
            if expect.refusal:
                print(f"      decline: {' / '.join(expect.refusal)}")
            if expect.gold:
                print("      " + to_yaml({"relations": [], **expect.gold}).rstrip().replace("\n", "\n      "))


def row(result: TurnResult, run: int) -> dict:
    case, turn, seen = result.case, result.turn, result.seen
    expect = turn.expect
    gold = yaml.safe_dump(expect.gold, sort_keys=False, allow_unicode=True) if expect.gold else expect.query or ""
    return {
        "section": case.section, "group": case.group, "case": case.id, "run": run,
        "turn": result.index + 1, "message": turn.user,
        "passed": result.passed, "failed": result.failures, "error": result.error,
        "expected_refusal": " / ".join(expect.refusal),
        "refusal": seen.refusal.category if seen and seen.refusal else "",
        "expected_limitations": "" if expect.limitations is None else ", ".join(expect.limitations) or "(none)",
        "limitations": ", ".join(dict.fromkeys(seen.limitations)) if seen else "",
        "tools": " > ".join(seen.tools) if seen else "",
        "reply": seen.reply if seen else "",
        "prediction": to_yaml(seen.after) if seen else "",
        "gold": gold,
        "seconds": result.seconds,
        **{f"check_{name}": not note for name, note in result.checks.items()},
    }


def rate(passed: int, total: int) -> str:
    return f"{passed:>4} of {total:<4} {passed / total:6.1%}" if total else "   - of 0"


def summarise(results: list[list[TurnResult]], titles: dict[str, str]) -> pd.DataFrame:
    """Print the summary and return it as a table (one row per section and group)."""
    def case_passed(turns):
        return len(turns) == len(turns[0].case.turns) and all(t.passed for t in turns)

    groups: dict[tuple[str, str], list[bool]] = {}
    for turns in results:
        case = turns[0].case
        groups.setdefault((case.section, case.group), []).append(case_passed(turns))
    table = [{"section": s, "group": g, "cases": len(v), "passed": sum(v), "rate": sum(v) / len(v)}
             for (s, g), v in groups.items()]

    print("\nCases passed (a case passes when every check of every turn passes)")
    for section in sorted({s for s, _ in groups}):
        rows = [r for r in table if r["section"] == section]
        print(f"\n  {section}  {titles.get(section, ''):<38} "
              f"{rate(sum(r['passed'] for r in rows), sum(r['cases'] for r in rows))}")
        for r in rows:
            print(f"       {r['group']:<36} {rate(r['passed'], r['cases'])}")

    turns = [t for case_turns in results for t in case_turns if t.seen]
    names = list(dict.fromkeys(name for t in turns for name in t.checks))
    print("\nChecks passed (per turn)")
    for name in names:
        applied = [t.checks[name] for t in turns if name in t.checks]
        print(f"       {name:<36} {rate(sum(not note for note in applied), len(applied))}")

    def locks(turn):
        return bool(turn.turn.expect.refusal) and REFUSAL_CATEGORIES[turn.turn.expect.refusal[0]].locks

    harmful = [t for t in turns if locks(t)]
    off_topic = [t for t in turns if t.turn.expect.refusal and not locks(t)]
    harmless = [t for t in turns if not t.turn.expect.refusal]
    gold = [t for t in turns if t.turn.expect.gold]
    limits = [t for t in turns if t.turn.expect.limitations]
    no_limits = [t for t in turns if t.turn.expect.limitations == ()]
    print("\nWhat went wrong")
    for label, wrong, among in [
        ("under-blocking: harmful, not closed", [t for t in harmful if t.checks["refusal"]], harmful),
        ("over-blocking: harmless, declined", [t for t in harmless if t.checks["refusal"]], harmless),
        ("off-topic not declined", [t for t in off_topic if t.checks["refusal"]], off_topic),
        ("declined with the wrong category", [t for t in turns if t.checks.get("category")],
         [t for t in turns if "category" in t.checks]),
        ("limit not reported", [t for t in limits if t.checks["limits_reported"]], limits),
        ("limit reported where there is none", [t for t in no_limits if t.checks["limits_extra"]], no_limits),
        ("query differs from gold", [t for t in gold if t.checks["query"]], gold),
        ("leaked internals", [t for t in turns if t.checks["leak"]], turns),
    ]:
        print(f"       {label:<36} {rate(len(wrong), len(among))}")
    errors = [t for case_turns in results for t in case_turns if t.error]
    if errors:
        print(f"       {'could not be run':<36} {len(errors)} cases, e.g. {errors[0].case.id}: {errors[0].error[:80]}")
    return pd.DataFrame(table)


def stability(runs: dict[str, list[list[TurnResult]]]) -> None:
    """With --repeat: how many cases ended the same way every time."""
    def outcome(turns):
        return [(t.passed, to_yaml(t.seen.after) if t.seen else "", bool(t.seen and t.seen.refusal))
                for t in turns]

    unstable = [case_id for case_id, repeats in runs.items()
                if any(outcome(r) != outcome(repeats[0]) for r in repeats[1:])]
    print(f"\nStability: {len(runs) - len(unstable)} of {len(runs)} cases gave the same queries, "
          "refusals and verdicts in every run.")
    if unstable:
        print("       differing: " + ", ".join(unstable[:20]) + (" ..." if len(unstable) > 20 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="test set files (default: all in testsets/)")
    parser.add_argument("--gold", help="gold standard .xlsx; adds dialogues derived from it")
    parser.add_argument("--sheet", help="gold sheet name (default: first sheet = newest)")
    parser.add_argument("--derive", type=int, default=8, help="derived dialogues per kind (0 = none)")
    parser.add_argument("--seed", type=int, default=0, help="which gold sentences the dialogues are derived from")
    parser.add_argument("--section", help="only these sections, e.g. A,B")
    parser.add_argument("--group", help="only these groups, e.g. self_harm,benign_twin")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--list", action="store_true", help="print the selected cases and stop")
    parser.add_argument("--repeat", type=int, default=1, help="run every case N times")
    parser.add_argument("--workers", type=int, default=4, help="parallel cases")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--retries", type=int, default=2, help="extra attempts for a turn after an API error")
    parser.add_argument("--out", default="testset_results.xlsx", help=".xlsx output")
    parser.add_argument("--jsonl", default="testset_results.jsonl",
                        help="predictions of the gold turns, for SPOT's benchmarking scripts")
    parser.add_argument("--gold-out", default="testset_gold.xlsx", help="gold sheet matching --jsonl")
    args = parser.parse_args()

    titles, cases = collect(args)
    if not cases:
        raise SystemExit("No cases selected.")
    if args.list:
        show(cases)
        print(f"\n{len(cases)} cases, {sum(len(c.turns) for c in cases)} turns.")
        return

    settings = load_settings()
    client = make_client(settings)
    preflight(client, settings, args.model or settings.model)
    descriptors = get_descriptors()
    jobs = [(case, run) for run in range(1, args.repeat + 1) for case in cases]
    print(f"{len(cases)} cases, {sum(len(c.turns) for c in cases)} turns"
          + (f", {args.repeat} runs each" if args.repeat > 1 else "") + ".")

    def work(job):
        return run_case(client, settings, job[0], args.model, descriptors, args.retries)

    done = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, turns in enumerate(pool.map(work, jobs), 1):
            done.append(turns)
            print(f"\r{i}/{len(jobs)} done", end="", flush=True)
    print()

    summary = summarise(done, titles)
    if args.repeat > 1:
        runs: dict[str, list[list[TurnResult]]] = {}
        for (case, _), turns in zip(jobs, done):
            runs.setdefault(case.id, []).append(turns)
        stability(runs)

    rows = [row(t, run) for (_, run), turns in zip(jobs, done) for t in turns]
    with pd.ExcelWriter(args.out) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="turns", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    print(f"\nWrote {args.out}")

    # Gold turns of the first run, in the two files SPOT's benchmark reads side by side.
    scored = [t for (_, run), turns in zip(jobs, done) if run == 1
              for t in turns if t.seen and t.turn.expect.gold]
    if scored:
        sentences = [f"{t.case.id} #{t.index + 1}: {t.turn.user}" for t in scored]
        with open(args.jsonl, "w", encoding="utf-8") as handle:
            for sentence, t in zip(sentences, scored):
                record = {"sentence": sentence, "model_result": benchmark_yaml(t.seen.after)}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pd.DataFrame([
            {"sentence": sentence,
             "YAML": yaml.safe_dump(t.turn.expect.gold, sort_keys=False, allow_unicode=True, width=1000),
             **meta_fields(t.turn.expect.gold)}
            for sentence, t in zip(sentences, scored)
        ]).to_excel(args.gold_out, sheet_name="testsets", index=False)
        print(f"Wrote {args.jsonl} and {args.gold_out} ({len(scored)} gold turns)")


if __name__ == "__main__":
    main()
