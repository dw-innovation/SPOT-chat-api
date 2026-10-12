<img width="1280" height="200" alt="Github-Banner_spot" src="https://github.com/user-attachments/assets/bec5a984-2f1f-44e7-b50d-cc6354d823cd" />

# 💬 SPOT Chat API

**Build a SPOT search query by chatting — and keep refining it.**
This module turns a conversation into a structured SPOT query (area, entities, relations) instead
of translating a single sentence in one shot. Users can correct, extend and narrow their search
over several messages: *"make it 200 m"*, *"add a bus stop nearby"*, *"search in Bonn instead"*.

The output is the same YAML the rest of SPOT already consumes, so it plugs into
[`osm-tag-search-api`](https://github.com/dw-innovation/kid2-spot-osm-tag-search-api) and
[`osm-query-api`](https://github.com/dw-innovation/kid2-spot-osm-query-api) unchanged.

---

## ✨ Key Features

- 💬 Multi-turn chat instead of one-shot translation — with undo and manual edits
- 🧰 The model never writes YAML directly: every change goes through validated tools
- 🏷️ Grounded in SPOT's tag bundles — object names, properties and colours are checked against the
  descriptor tables and [SPOT's tag search](https://tags.findthatspot.io/)
- 📏 Relative spatial terms ("next to", "nearby") resolved from SPOT's official lookup table
- 🤖 Works with any OpenAI-compatible endpoint (GPT-OSS, GPT-5, local Ollama, …) — no fine-tuning
- 🐳 Stateless, Dockerized HTTP API
- 📊 Benchmark runner against the gold standard, exporting SPOT's JSONL result format

---

## 📦 Architecture Overview

```
kid2-spot-chat/
├── api.py                  # HTTP API (FastAPI): /chat, /validate, /health
├── app.py                  # Gradio demo UI (Hugging Face Space)
├── spot_chat/
│   ├── agent.py            # chat loop: LLM → tool calls → reply
│   ├── tools.py            # the 13 tools the model may call
│   ├── policy.py           # what is declined, and what the query cannot express
│   ├── query.py            # validation & normalisation of the query
│   ├── descriptors.py      # SPOT tag bundles: lookup, message scan, suggestions
│   ├── tag_search.py       # client for the SPOT tag search API
│   ├── prompts.py          # system prompt (partly generated from the tables)
│   ├── schema.py           # units, spatial terms, operators, limits
│   └── compare.py          # scoring against the gold standard
├── data/                   # tag bundles + colour table
├── evaluate.py             # benchmark runner
├── testsets/               # behaviour test sets: refusals, injection, multi-turn, limits
├── run_testsets.py         # runs the test sets against the model
├── Dockerfile / docker-compose.yml
└── API.md                  # request/response reference for frontend devs
```

---

## 🚀 Quickstart

### With Docker

```bash
git clone https://github.com/dw-innovation/kid2-spot-chat.git
cd kid2-spot-chat
cp .env.example .env        # fill in LLM_API_KEY and LLM_BASE_URL
docker compose up --build   # http://localhost:8000
```

Interactive API docs: `http://localhost:8000/docs` (start with `SPOT_DEBUG=1`)

### Locally

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn api:app --reload        # API on http://localhost:8000
```

```bash
pip install "gradio==6.28.0"
python app.py                   # demo UI on http://localhost:7860
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill it in.

### Language model

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key of the OpenAI-compatible endpoint. |
| `LLM_BASE_URL` | Endpoint URL, e.g. `https://…/v1` or `http://localhost:11434/v1` for Ollama. |
| `LLM_MODEL` | Model name (default `GPT-oss-120b`). |
| `LLM_MODELS` | Optional comma-separated list; adds a model picker to the demo UI. |
| `LLM_TEMPERATURE` | Optional, `0` recommended for reproducible benchmarks. |
| `LLM_REASONING_EFFORT` | Optional, `low`/`medium`/`high` for reasoning models. |
| `LLM_MAX_TOOL_ROUNDS` | Tool-call rounds per message (default `8`). |
| `LLM_TIMEOUT` | Request timeout in seconds (default `120`). |

### SPOT tables & services

| Variable | Description |
|----------|-------------|
| `SPOT_TAG_SEARCH_URL` | Tag search endpoint, default `https://tags.findthatspot.io/api/search`; empty disables it. |
| `SPOT_BUNDLES_PATH` | Tag-bundle spreadsheet, default `data/SPOT_OSM-tag-bundles.xlsx`. |
| `SPOT_COLOURS_PATH` | Colour table, default `data/spot_colours.csv`. |
| `SPOT_MAX_ENTITIES`, `SPOT_MAX_PROPERTIES` | Query size SPOT is trained for (default `3` each). |
| `SPOT_INCLUDE_SPATIAL_TERM` | `1` also writes the spatial term ("next to") into the YAML; by default only its distance. |

> Keep `data/SPOT_OSM-tag-bundles.xlsx` in sync with the table behind the tag search API. Names
> the file does not contain cannot be suggested to the model.

### API server

| Variable | Description |
|----------|-------------|
| `SPOT_API_KEY` | If set, every request needs the header `X-API-Key: <value>`. |
| `SPOT_CORS_ORIGINS` | Comma-separated origins for browser calls (default `*`). |
| `SPOT_RATE_LIMIT` | `/chat` calls per minute and client address (default `30`, `0` = off). |
| `SPOT_DEBUG` | `1` enables `/docs`, fills `tool_calls` and makes `/health` detailed. Leave unset in production. |

Requests are capped (message 2,000 characters, 200 history messages, 256 kB), `history` takes
only `user` and `assistant` turns, and `model` must be one of `LLM_MODELS`. See "Going public"
in [API.md](./API.md) for what a public deployment has to add.

---

## 🔌 API

**The server keeps no state.** Every call carries the current query and the chat history, and gets
the updated ones back. Keep both in the frontend and send them again with the next message, so
restarts and multiple instances lose nothing — and users can edit the query in between.

### `POST /chat`

```jsonc
{
  "message": "show me all bars next to rivers in cologne",  // required
  "query": null,        // query from the previous response; null starts a new search
  "query_yaml": null,   // alternative to "query": the same as a YAML string
  "history": [],        // previous turns: {"role": "user"|"assistant", "content": "..."}
  "model": null         // optional model override
}
```

```jsonc
{
  "reply": "Bars within 50 m of a river in Cologne. I used 50 m for \"next to\".",
  "query": {
    "area": {"type": "area", "value": "Cologne"},
    "entities": [
      {"id": 0, "name": "bar", "type": "nwr"},
      {"id": 1, "name": "river", "type": "nwr"}
    ],
    "relations": [
      {"source": 0, "target": 1, "type": "distance", "value": "50 m"}
    ]
  },
  "query_yaml": "area:\n  type: area\n  value: Cologne\n…",
  "changed": true,     // false if the turn changed nothing (e.g. the user asked a question)
  "history": [ … ],    // send this back with the next message
  "tool_calls": [      // what the model did — only with SPOT_DEBUG=1, otherwise []
    {"name": "new_query", "ok": true, "message": "Started a new query with 2 entities and 1 relations.",
     "arguments": { … }, "warnings": [], "notes": []}
  ],
  "refusal": null,     // set when the message was declined: {"category", "locked", "url"}
  "limitations": []    // parts of the request the query cannot express
}
```

Show `reply` in the chat, draw the map from `query`, and redraw when `changed` is `true`.

The assistant declines messages that are not about map search, and searches with a stated
harmful purpose. After a harmful request the chat stays closed until it is reset (`history: []`);
the server enforces this through a marker in the `history` it returns, so send `history` back
unchanged. See **[API.md](./API.md)** for what to render in each case.

```js
const res = await fetch(`${CHAT_API}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, query: state.query, history: state.history }),
});
const data = await res.json();
state.query = data.query;      // for the next call and for the map
state.history = data.history;
```

### `POST /validate`

Checks and normalises a query the user edited by hand or in a graph editor, without calling the
model (`"400 meters"` → `"400 m"`, entity ids renumbered). Returns
`{"valid": true, "query": …, "query_yaml": …}` or `{"valid": false, "error": "…"}`.

### `GET /health`

`{"status": "ok"}`; with `SPOT_DEBUG=1` also the model and what is loaded.

### Status codes

| Code | Meaning |
|------|---------|
| `200` | fine |
| `400` | empty message, unknown `model`, or the query sent is invalid (`detail` explains) |
| `401` | `SPOT_API_KEY` is set and `X-API-Key` is missing or wrong |
| `413` / `422` | the request is too large or a field has the wrong shape |
| `429` | too many messages from this client (`Retry-After` says how long to wait) |
| `502` | the language model could not be reached (the reason is in the server log) |

A turn takes a few seconds because the model may call several tools, so allow a generous client
timeout and show a typing indicator. Full reference with more examples: **[API.md](./API.md)**.

---

## 🧠 How a message is processed

1. The message goes to the model together with the current query, the recent history, and the
   SPOT vocabulary found in the message (typos corrected).
2. The model calls tools: `new_query` for a new search, or small editing tools
   (`add_entities`, `update_relation`, `remove_properties`, …).
3. Each call is validated in plain Python: no OSM tags as names, locations belong in the area,
   property types follow the tag table, distances are normalised (`"1,200 meters"` → `"1200 m"`),
   spatial terms get their distance from SPOT's lookup table, cluster distances and entity ids are
   filled in. A call that breaks a rule changes nothing and comes back with an explanation.
4. Names are checked against the descriptor tables and the tag search API. The model is told what
   SPOT would search for, or which descriptor the user's own wording matches.
5. The model fixes what the tools reported, then answers in one or two sentences.

---

## 📊 Benchmarking

```bash
python evaluate.py goldstandard_testing_dataset.xlsx --workers 4
```

Runs every sentence of the gold standard as a fresh chat and writes

- `evaluation_results.xlsx` — per sentence: prediction, gold, what differed, tools used
- `evaluation_results.jsonl` — `{"sentence": …, "model_result": …}` for SPOT's benchmarking scripts

Useful flags: `--limit` for a quick check, `--model` to compare models, `--retries` for sentences
that produce no YAML, `--sheet` to pick a gold sheet.

---

## 🛡️ Behaviour test sets

The gold standard only has one-shot sentences. What the bot declines, how it handles prompt
injection, several turns and requests the query cannot express is tested with the sets in
`testsets/`:

| File | Tests |
|---|---|
| `safety.yaml` | harmful requests per category, their harmless twins, off-topic, abuse |
| `injection.yaml` | injected instructions in the message, the query, the history and tag search results |
| `dialogues.yaml` | edits, references, corrections, manual edits, long chats, safety across turns |
| `limits.yaml` | unsupported requests: the supported part is built, the rest is reported |

```bash
python run_testsets.py --workers 4                                   # all four sets
python run_testsets.py --gold goldstandard_testing_dataset.xlsx      # + dialogues derived from the gold standard
python run_testsets.py --section A --group self_harm,benign_twin     # a part of them
python run_testsets.py --gold goldstandard_testing_dataset.xlsx --list   # read the cases, call no model
python run_testsets.py --repeat 3                                    # is the result the same every time?
```

Each case is played like a chat through the API and every turn is checked. The summary gives
the cases passed per group, and separately under-blocking (a harmful request was answered),
over-blocking (a harmless one was declined), limits missed or reported without reason, and
queries that differ from their gold. Output:

- `testset_results.xlsx` — one row per turn: what was expected, what happened, which check failed
- `testset_results.jsonl` + `testset_gold.xlsx` (sheet `testsets`) — the turns that have a gold
  query, as a prediction file and a gold sheet for SPOT's benchmarking scripts

With `--gold`, more dialogues are derived from the gold sentences: the chat starts with the gold
query and asks for one mechanical edit (new distance, new area, drop an object, ...), or builds
the gold query one object per message. The expected query is computed by the tools; only the
wording comes from templates in `testsets/derive.py`. The case format is described at the top of
`testsets/cases.py`.

---

## 🧪 Tests

```bash
python -m pytest -q
```

Covers the tools, validation, the chat loop, the tag search integration and the API, and that the
behaviour test sets load and are scored as described (without calling the model). The gold
round-trip test additionally needs `goldstandard_testing_dataset.xlsx` in the repository root.

---

## 🧩 Part of the SPOT System

This module replaces the one-shot sentence → YAML step of
[`central-nlp-api`](https://github.com/dw-innovation/kid2-spot-central-nlp-api) with a
conversation. It produces the same YAML, so tag search, OSM query and the map frontend stay as
they are.

---

## 🔗 Related Docs

- [Main SPOT Repo](https://github.com/dw-innovation/kid2-spot)
- [Frontend UI](https://github.com/dw-innovation/kid2-spot-frontend)
- [Central NLP API](https://github.com/dw-innovation/kid2-spot-central-nlp-api)
- [OSM Tag Search API](https://github.com/dw-innovation/kid2-spot-osm-tag-search-api)
- [OSM Query API](https://github.com/dw-innovation/kid2-spot-osm-query-api)
- [Tag bundles viewer & editor](https://tags.findthatspot.io/)

---

## 🙌 Contributing

We welcome contributions of all kinds — from developers, journalists, mappers, and more!
See [CONTRIBUTING.md](https://github.com/dw-innovation/kid2-spot/blob/main/CONTRIBUTING.md) for how
to get started. Also see our [Code of Conduct](https://github.com/dw-innovation/kid2-spot/blob/main/CODE_OF_CONDUCT.md).

---

## 📜 License

Licensed under [AGPLv3](../LICENSE).

© Deutsche Welle Research & Cooperation Projects
