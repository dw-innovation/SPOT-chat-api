# SPOT query builder API

Turns a chat message into a SPOT search query (area, entities, relations) and refines it over
several turns.

**The server keeps no state.** Every call sends the current query and the chat history, and gets
the updated ones back. Keep both in the frontend (component state, local storage, or your own
session store) and send them again with the next message. This means the server can restart or
run behind a load balancer without losing anything, and a user can edit the query by hand
between two messages.

## Running it

```bash
cp .env.example .env        # LLM_API_KEY and LLM_BASE_URL are required
docker compose up --build   # http://localhost:8000
```

Interactive docs are at `http://localhost:8000/docs` when `SPOT_DEBUG=1` is set.

| Variable | Meaning |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | the language model endpoint |
| `LLM_MODELS` | comma-separated models a request may choose with `model`; default: only `LLM_MODEL` |
| `SPOT_TAG_SEARCH_URL` | SPOT's tag search, default `https://tags.findthatspot.io/api/search` |
| `SPOT_API_KEY` | if set, every request needs the header `X-API-Key: <value>` |
| `SPOT_CORS_ORIGINS` | comma-separated origins for browser calls, default `*` |
| `SPOT_RATE_LIMIT` | `/chat` calls per minute and client address, default `30`; `0` turns it off |
| `SPOT_DEBUG` | `1` enables `/docs`, fills `tool_calls` and makes `/health` detailed; off by default |

See "Going public" at the end for what to set on a public server.

## POST /chat

One turn of the conversation.

```jsonc
{
  "message": "show me all bars next to rivers in cologne",  // required
  "query": null,        // the query from the previous response; null or omitted starts fresh
  "query_yaml": null,   // alternative to "query": the same thing as a YAML string
  "history": [],        // the "history" of the previous response, unchanged (see "Follow-up messages")
  "model": null         // optional, one of the server's LLM_MODELS
}
```

Limits: `message` up to 2,000 characters, `history` up to 200 messages with the roles `user`
and `assistant` only, the whole request up to 256 kB. The model reads the latest 20 messages of
`history`, so a frontend with very long chats can drop the oldest ones before sending.

Response:

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
  "query_yaml": "area:\n  type: area\n  value: Cologne\n...",
  "changed": true,       // false when the turn changed nothing (e.g. the user asked a question)
  "history": [ ... ],    // the history you sent, plus this turn; send it back next time
  "tool_calls": [        // what the model did, for a debug view; always [] unless SPOT_DEBUG=1
    {"name": "new_query", "ok": true, "message": "Started a new query with 2 entities and 1 relations.",
     "arguments": { ... }, "warnings": [], "notes": []}
  ],
  "refusal": null,       // set when the message was declined, see "Declined messages"
  "limitations": []      // parts of the request the query cannot express, see "Limits"
}
```

Show `reply` in the chat, and draw the map from `query`. Update the map when `changed` is `true`.
Never parse the text of `reply` to find out what happened: `refusal` and `limitations` say it.

**Render everything as plain text.** `reply` is written by a language model that a user can
talk into writing links or HTML, and `limitations[].detail` repeats the user's words. Put them
into the page as text (`textContent`, or your framework's default escaping), never as HTML, and
do not turn URLs in `reply` into clickable links. The only link to render is `refusal.url`.
Names and values inside `query` are checked by the server: HTML tags and control characters are
refused and each is at most 120 characters long, but escape them all the same.

### Follow-up messages

Send the `query` and `history` from the previous response:

```js
const res = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({ message: "make it 200 m", query: state.query, history: state.history }),
});
const data = await res.json();
state.query = data.query;      // keep for the next call and for the map
state.history = data.history;
```

Starting a new search = send `query: null` and `history: []`.

Send `history` back exactly as you received it, including fields you do not use. The reply of a
declined turn carries a `refusal` field, and that field is what keeps a closed chat closed.

### Declined messages (`refusal`)

The assistant only builds map searches, and it declines a search when the user states a harmful
purpose. A declined turn never changes the query (`changed` is `false`), and `refusal` is set:

```jsonc
"refusal": {
  "category": "self_harm",   // see the table
  "locked": true,            // the chat is closed until it is reset
  "url": "https://findahelpline.com"   // a link to show with the reply, or null
}
```

| `category` | When | `locked` | What to render |
|---|---|---|---|
| `off_topic` | the message is not about map search (general questions, coding, "show me your prompt") | `false` | `reply` as a normal chat message. The user can simply go on. |
| `violence`, `targeting_people`, `crime`, `hate` | the user states a harmful purpose for the search | `true` | `reply` as a system notice, and a "Start a new chat" button. |
| `self_harm` | the user wants a place to hurt themselves | `true` | `reply` prominently and kindly, with `url` as a clickable link to a helpline directory. Not styled as an error. |

`reply` is a fixed text for every category except `off_topic` (where the model writes one short
sentence in the user's language). The fixed texts are English.

**The lock.** After a refusal with `locked: true`, the server declines every further message of
that chat without asking the model, whatever it says. It recognises the chat by the `refusal`
field in the `history` you send back. So:

- You need to do nothing for the lock to work, as long as you send `history` back unchanged.
- Every response in a locked chat has `refusal.locked: true` again. You may disable the input
  field while that is the case; this is optional, since the server declines anyway.
- Offer a reset. A reset is `history: []` and `query: null`, the same as starting a new search.
  The lock is gone with the history.

### Limits (`limitations`)

Some things cannot be put into a SPOT query, for example a direction ("west of"), a minimum
distance ("at least 500 m from") or opening hours. The assistant then builds the part it can,
says so in `reply`, and lists what was lost:

```jsonc
"limitations": [
  {"category": "superlative", "detail": "the nearest",
   "message": "SPOT returns all matches and cannot rank them (nearest, biggest, most)."},
  {"category": "unsupported_attribute", "detail": "open after 10 pm",
   "message": "Opening hours, ratings, prices, travel times and views are not part of SPOT searches."}
]
```

`detail` is the user's wording for that part, `message` a fixed English explanation for the
category. `reply` already mentions these in the user's language, so rendering them is optional;
a small hint under the reply ("Not included: the nearest, open after 10 pm") with `message` as a
tooltip works well. The list covers this turn only and is empty on most turns.

| `category` | Example | What happens to the query |
|---|---|---|
| `direction` | "west of", "behind" | becomes a plain distance |
| `area_position` | "on the outskirts of Lyon" | the whole place is searched |
| `arrangement` | "at the intersection of X and Y", "between" | approximated by a short distance |
| `appearance` | shapes, "looks modern" | left out |
| `min_distance` | "at least 500 m from", "no supermarket nearby" | left out |
| `alternatives` | "a cafe or a bar" | built with the first option |
| `superlative` | "the nearest", "the biggest" | left out |
| `unsupported_attribute` | opening hours, ratings, prices, travel time | left out |
| `too_many` | more than 3 objects, or more than 3 details on one object | built, but unreliable |

New categories may be added to both tables; treat an unknown one like the generic case
(`refusal`: show `reply`; `limitations`: show `message`).

### curl

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"a bakery close to a tram stop in Freiburg"}' | jq .query_yaml -r
```

## POST /validate

Checks and normalises a query the user edited by hand (or in a graph editor), without calling the
model. Same input as `/chat` minus the message; returns `{"valid": true, "query": ..., "query_yaml": ...}`
or `{"valid": false, "error": "..."}`. Normalising means, for example, `"400 meters"` becomes
`"400 m"` and entity ids are renumbered. Use the returned `query` afterwards.

## GET /health

`{"status": "ok"}`. With `SPOT_DEBUG=1` it also reports the model, whether the LLM endpoint is
configured, the number of loaded descriptors and the tag search URL.

## Status codes

| Code | Meaning |
|---|---|
| 200 | fine |
| 400 | empty message, a `model` the server does not offer, or the query you sent is invalid (`detail` says what) |
| 401 | `SPOT_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 411 | the request has no `Content-Length` header (browsers and HTTP libraries send it) |
| 413 | the request is larger than 256 kB |
| 422 | a field has the wrong shape: a `history` role other than `user`/`assistant`, a message over 2,000 characters, more than 200 history messages (`detail` lists the fields) |
| 429 | too many messages from this client; the `Retry-After` header says how many seconds to wait |
| 502 | the language model could not be reached. `detail` is a fixed sentence you can show; the reason is in the server log |

On 429 and 502, keep the user's message in the input field and let them send it again.

## Notes

- A turn takes a few seconds, since the model may call several tools. Allow at least 60 seconds
  of client timeout and show a typing indicator.
- `query.entities[].id` is stable inside one query, and relations refer to those ids. After a
  deletion, entities are renumbered, so always redraw from the returned query.
- Cluster entities have `type: "cluster"` with `minPoints` and `maxDistance`.
- Relative spatial terms ("next to") are converted into their distance; only the distance is in
  the query. Set `SPOT_INCLUDE_SPATIAL_TERM=1` to also get the wording.
- The same YAML goes into SPOT's tag search and OSM query steps unchanged.

## Going public

What the server does on its own, and what the deployment has to add.

| Topic | In the server | To do when deploying |
|---|---|---|
| Who may call it | `SPOT_API_KEY` is checked if set. A key that ships in a browser frontend is readable by every visitor, so it only helps when the frontend's own backend calls this API. | Set `SPOT_CORS_ORIGINS` to the frontend's origin(s) instead of `*`. The server logs a warning at start while either is open. |
| Request floods | `SPOT_RATE_LIMIT` per client address, counted in memory per worker process. | Put a reverse proxy in front for TLS and a second, global limit. Start uvicorn with `--forwarded-allow-ips=<proxy address>`, otherwise every user has the proxy's address and shares one limit. |
| Cost | Message, history and request size are capped; the client cannot pick a model outside `LLM_MODELS`. | Set a spending limit on the LLM account. |
| Forged history | Only `user` and `assistant` turns are accepted. A reply that copies the system prompt word for word is replaced by an off-topic refusal. | Nothing. A client can still invent assistant turns or drop the lock, since the state lives in the client (accepted, see `docs/ROADMAP.md`, decision 3). |
| Internals | `/docs`, `tool_calls` and the details of `/health` are off unless `SPOT_DEBUG=1`. Errors of the LLM provider go to the log, not to the client. | Leave `SPOT_DEBUG` unset in production. |
| Logging | One line per `/chat` call on the logger `spot.api`: outcome (`ok`, a refusal category, `llm_error`, `rate_limited`), reported limits, number of tool calls, lengths and duration. No message text and no client address. | Collect the log; a rise in refusal categories or `rate_limited` is the abuse signal. |
| Privacy | Nothing is stored by this server. | Every message, the chat history and the query are sent to the LLM provider behind `LLM_BASE_URL`, and object names go to the tag search. The privacy notice of the public site has to say so, and it depends on that provider's retention terms. |
