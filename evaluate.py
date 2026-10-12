"""Benchmark the chat pipeline against the gold-standard sentences.

Every sentence is sent as the first message of a fresh chat, exactly like in the
demo, and the resulting query is compared with the gold YAML.

    python evaluate.py goldstandard_testing_dataset.xlsx                 # newest sheet, all rows
    python evaluate.py data.xlsx --limit 30 --workers 4 --model gpt-5.5
    python evaluate.py data.xlsx --sheet gold_annotations_14072025 --out results.xlsx

Writes one row per sentence (prediction, gold, per-aspect scores, notes) to
--out and prints a summary. Uses the same LLM settings as the app (.env).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd
import yaml

from spot_chat.agent import ErrorEvent, ReplyEvent, ToolEvent, run_turn
from spot_chat.compare import compare
from spot_chat.config import ENV_FILE, load_settings, make_client
from spot_chat.descriptors import get_descriptors
from spot_chat.query import empty_query, to_export, to_yaml

ASPECTS = ["area", "entity_count", "entity_names", "entity_bundles", "entity_types", "properties",
           "relation_structure", "relations", "exact", "exact_literal"]


def load_gold(path: str, sheet: str | None) -> pd.DataFrame:
    sheet = sheet or pd.ExcelFile(path).sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[df["sentence"].notna() & df["YAML"].notna()].copy()
    df["gold"] = df["YAML"].map(yaml.safe_load)
    print(f"Loaded {len(df)} sentences from sheet '{sheet}'.")
    return df.reset_index(drop=True)


def benchmark_yaml(query: dict) -> str:
    """YAML in the format of SPOT's benchmarking JSONL files (keys sorted); "" if no query was built."""
    if not query["entities"]:
        return ""
    return yaml.safe_dump(to_export(query), sort_keys=True, allow_unicode=True, width=1000)


NUDGE = ("Build the query now with new_query, using your best interpretation of my request. "
         "Do not ask a question first.")


def run_sentence(client, settings, sentence: str, model: str | None, force_tool: bool,
                 retries: int = 3) -> dict:
    """Run one sentence like a fresh chat. If no YAML comes out, try again (at most `retries` times):
    after an API error or crash it waits and starts over; after a text-only answer (e.g. a
    question) it continues the chat and asks the model to build its best interpretation."""
    query, tools, errors, reply, reasons = empty_query(), [], 0, "", []
    conversation, message = [], sentence
    started = time.time()
    for attempt in range(1, retries + 2):
        turn_reply = ""
        try:
            for event in run_turn(client, settings, conversation, message, query,
                                  model=model, require_tool=force_tool):
                query = event.query
                if isinstance(event, ToolEvent):
                    tools.append(event.result.name if event.result.ok else f"{event.result.name}(error)")
                    errors += not event.result.ok
                elif isinstance(event, ReplyEvent):
                    turn_reply = event.text
                elif isinstance(event, ErrorEvent):
                    turn_reply = f"ERROR: {event.text}"
        except Exception as exc:  # one broken sentence must not stop the run
            turn_reply = f"CRASH: {exc!r}"
        reply = turn_reply
        if query["entities"] or attempt > retries:
            break
        if not turn_reply or turn_reply.startswith(("ERROR", "CRASH")):
            reasons.append("api error")
            time.sleep(min(30, 2 ** attempt))           # API trouble: wait, then start over
            conversation, message = [], sentence
        else:
            reasons.append("text reply")                 # model answered instead of building
            conversation = conversation + [{"role": "user", "content": message},
                                           {"role": "assistant", "content": turn_reply}]
            message = NUDGE
    return {
        "query": query, "tools": tools, "tool_errors": errors, "reply": reply,
        "attempts": attempt, "retry_reasons": ", ".join(reasons),
        "seconds": round(time.time() - started, 1),
    }


def preflight(client, settings, model: str) -> None:
    """One tiny request before the run, so a setup problem shows up once, clearly."""
    print(f".env file:  {ENV_FILE if ENV_FILE.exists() else f'NOT FOUND (expected at {ENV_FILE})'}")
    print(f"endpoint:   {settings.base_url or 'LLM_BASE_URL is not set, so the OpenAI API would be used'}")
    print(f"model:      {model}")
    print(f"API key:    {'set' if settings.api_key else 'MISSING (LLM_API_KEY)'}")
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with OK."}],
            **settings.completion_kwargs(),
        )
    except openai.OpenAIError as exc:
        raise SystemExit(
            f"\nThe model could not be reached: {exc}\n\n"
            "What the error usually means:\n"
            "  401 / 403        the API key is wrong, or missing in .env\n"
            "  404              wrong model name, or LLM_BASE_URL is missing its /v1 ending\n"
            "  Connection error LLM_BASE_URL is wrong, or the endpoint is only reachable from\n"
            "                   certain networks (VPN, firewall, IP allow-list)\n"
            "  400 mentioning temperature / reasoning_effort\n"
            "                   the endpoint does not accept that setting; remove it from .env"
        ) from None
    print("connection: OK\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("excel", help="gold standard .xlsx file")
    parser.add_argument("--sheet", help="sheet name (default: first sheet = newest)")
    parser.add_argument("--limit", type=int, help="only the first N sentences")
    parser.add_argument("--workers", type=int, default=4, help="parallel requests")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--force-tool", action="store_true",
                        help="send tool_choice=required on the first call (if the server supports it)")
    parser.add_argument("--retries", type=int, default=3,
                        help="extra attempts for a sentence that produced no YAML (0 = none)")
    parser.add_argument("--out", default="evaluation_results.xlsx", help=".xlsx or .csv output")
    parser.add_argument("--jsonl", default="evaluation_results.jsonl",
                        help="predictions as JSONL for SPOT's benchmarking scripts")
    args = parser.parse_args()

    settings = load_settings()
    client = make_client(settings)
    preflight(client, settings, args.model or settings.model)
    df = load_gold(args.excel, args.sheet)
    if args.limit:
        df = df.head(args.limit)

    def work(sentence):
        return run_sentence(client, settings, sentence, args.model, args.force_tool, args.retries)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = []
        for i, run in enumerate(pool.map(work, df["sentence"]), 1):
            runs.append(run)
            print(f"\r{i}/{len(df)} done", end="", flush=True)
    print()

    descriptors = get_descriptors()
    rows = []
    for (_, gold_row), run in zip(df.iterrows(), runs):
        result = compare(to_export(run["query"]), gold_row["gold"], descriptors)
        rows.append({
            "sentence": gold_row["sentence"],
            **{k: v for k, v in result.as_dict().items() if k != "notes"},
            "first_tool_new_query": bool(run["tools"]) and run["tools"][0] == "new_query",
            "tools": " > ".join(run["tools"]),
            "tool_errors": run["tool_errors"],
            "notes": result.notes,
            "prediction": to_yaml(run["query"]),
            "gold": yaml.safe_dump(gold_row["gold"], sort_keys=False, allow_unicode=True),
            "reply": run["reply"],
            "asked_question": "?" in run["reply"] and not run["reply"].startswith(("ERROR", "CRASH")),
            "empty_result": not run["query"]["entities"],
            "attempts": run["attempts"],
            "retry_reasons": run["retry_reasons"],
            "seconds": run["seconds"],
        })
    out = pd.DataFrame(rows)

    print(f"\nModel: {args.model or settings.model}   sentences: {len(out)}")
    for aspect in ASPECTS + ["first_tool_new_query"]:
        print(f"  {aspect:<22} {out[aspect].mean():6.1%}")
    print(f"  {'tool errors (total)':<22} {int(out['tool_errors'].sum())}")
    print(f"  {'asked a question':<22} {int(out['asked_question'].sum())} "
          f"(of which without YAML: {int((out['asked_question'] & out['empty_result']).sum())})")
    retried = out["attempts"] > 1
    print(f"  {'needed a retry':<22} {int(retried.sum())} "
          f"(api error: {int(out['retry_reasons'].str.contains('api error').sum())}, "
          f"text reply: {int(out['retry_reasons'].str.contains('text reply').sum())})")
    print(f"  {'no YAML produced':<22} {int(out['empty_result'].sum())} (after all retries)")
    print(f"  {'avg seconds':<22} {out['seconds'].mean():.1f}")

    with open(args.jsonl, "w", encoding="utf-8") as handle:
        for sentence, run in zip(df["sentence"], runs):
            record = {"sentence": sentence, "model_result": benchmark_yaml(run["query"])}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {args.jsonl}")

    if args.out.endswith(".csv"):
        out.to_csv(args.out, index=False)
    else:
        out.to_excel(args.out, index=False)
    summary = {a: round(float(out[a].mean()), 4) for a in ASPECTS}
    print(f"\nWrote {args.out}\n{json.dumps(summary)}")


if __name__ == "__main__":
    main()
