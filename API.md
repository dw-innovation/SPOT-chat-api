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
| `SPOT_RATE_LIMIT` | `/chat` calls per minute, default `30`; `0` turns it off. Counted per user if the caller sends the key and an `X-Spot-User` header, otherwise per client address |
| `LLM_TURN_TIMEOUT` | seconds one `/chat` call may take before it ends with 504; default off. Set it below the timeout of whatever calls this API (`50` for the SPOT frontend). While it is set, failed model calls are not retried automatically |
| `SPOT_DEBUG` | `1` enables `/docs`, fills `tool_calls` and makes `/health` detailed; off by default |

See "Going public" at the end for what to set on a public server, and "Inside the SPOT
frontend" for how this API fits into the existing SPOT system.

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
If you render `reply` as Markdown because the lists and bold text read better, switch off links
and images in the renderer (and raw HTML, if it is not off by default): an image is fetched from
its address the moment the reply is shown.
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

Every error has the same body, in the shape the other SPOT services use:

```jsonc
{"status": "error",
 "message": "Too many messages. Please wait a moment.",   // always one string, safe to show
 "detail": "Too many messages. Please wait a moment."}    // FastAPI's field: the same string, or a list of fields for 422
```

| Code | Meaning |
|---|---|
| 200 | fine |
| 400 | empty message, a `model` the server does not offer, or the query you sent is invalid (`message` says what) |
| 401 | `SPOT_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 411 | the request has no `Content-Length` header (browsers and HTTP libraries send it) |
| 413 | the request is larger than 256 kB |
| 422 | a field has the wrong shape: a `history` role other than `user`/`assistant`, a message over 2,000 characters, more than 200 history messages (`message` names the fields in one line, `detail` lists them) |
| 429 | too many messages from this user or client; the `Retry-After` header says how many seconds to wait |
| 502 | the language model could not be reached. `message` is a fixed sentence you can show; the reason is in the server log |
| 504 | the turn took longer than `LLM_TURN_TIMEOUT`. `message` is a fixed sentence; nothing was changed |

On 429, 502 and 504, keep the user's message in the input field and let them send it again.

## Notes

- A turn takes a few seconds, since the model may call several tools. Allow at least 60 seconds
  of client timeout and show a typing indicator. Without `LLM_TURN_TIMEOUT` a turn has no upper
  limit of its own (up to 8 model calls of up to 120 seconds each).
- `query.entities[].id` is stable inside one query, and relations refer to those ids. After a
  deletion, entities are renumbered, so always redraw from the returned query.
- Cluster entities have `type: "cluster"` with `minPoints` and `maxDistance`.
- Relative spatial terms ("next to") are converted into their distance; only the distance is in
  the query. Set `SPOT_INCLUDE_SPATIAL_TERM=1` to also get the wording.
- The query names objects and properties; it has no OSM tags. Turning it into the query the
  OSM query API runs (`nodes` with `filters`, `edges`) is a tag search lookup per name, which the
  SPOT frontend does itself (see the next section).

## Inside the SPOT frontend

How this API fits into the running SPOT system, from reading the SPOT repositories on
2026-10-12 (`dw-innovation/SPOT-frontend` and the three backend services; nothing was run).
The frontend's branch `feat/spot-chat-api` already calls `/chat`. As pushed on 2026-10-07 it
is an older build, written against this API as it was before `refusal`, `limitations` and the
limits existed. Newer frontend code for the current API exists but was not on GitHub when this
was written, so everything below that describes the chat branch, and above all the list at the
end, is a checklist to go through against the current frontend code, not a list of known bugs.

**The path of a message.** The browser never calls a backend. It posts to the frontend's own
route `/api/transformSentence`, which requires a signed-in user (NextAuth) and runs on the
frontend's server (Vercel). With `SPOT_CHAT_API` set, that route posts
`{message, query, history}` to `/chat` with the header `X-API-Key: $SPOT_CHAT_API_KEY`, and
otherwise forwards the sentence to the Central NLP API as before. So:

- `SPOT_API_KEY` is real protection here: it never reaches a browser. The user's session token
  is not sent to this API; checking it would mean sharing the frontend's `NEXTAUTH_SECRET`,
  with which it is encrypted.
- No browser calls this API, so CORS can be closed.
- Every request arrives from the frontend's server. The client address says nothing about the
  user, so the rate limit counts per `X-Spot-User` when that header comes with the key. The
  frontend already has the right value: `anonymizeUser(session)`, the pseudonym it sends to the
  NLP API as `username`. Without the header all users share one limit of 30 messages a minute.
- The route may run for 60 seconds (`maxDuration`); after that the platform ends it and the
  user sees "UnknownError". Run this API with `LLM_TURN_TIMEOUT=50` so that a slow turn ends
  with a 504 and a sentence instead.
- The frontend does not send `model`; the server's `LLM_MODEL` decides.

**From `query` to the query that is run.** `/chat` returns objects and properties by name. The
frontend converts them (`src/lib/chatQuery.server.ts`): one tag search lookup per entity and
per property (`limit=1`), the same steps as `adopt_generation.py` in the NLP API, and returns
the result as `imr` next to the fields of `/chat`. `imr` is what the query panel edits and what
goes to the OSM query API (`/validate-spot-query`, then `/run-spot-query`). Two consequences:

- A name the tag search has nothing for fails the conversion with a 400 after `/chat` has
  answered, and the turn is lost. This API does not refuse such a name. It only tells the
  model when it does not find one: object names in its local table or the tag search, property
  names in its local table (`data/SPOT_OSM-tag-bundles.xlsx`), which can differ from what the
  tag search returns.
- `/validate` here and `/validate-spot-query` there check different things. Ours checks the
  chat query (names, distances, at most 3 objects, ids) and normalises it. Theirs checks the
  converted query against a JSON schema, and that every node has a filter, ids and names are
  unique, and no relation is doubled or points at itself. It accepts a query without nodes.

**State.** The frontend keeps `history` and the last `query` in memory (`useChatStore`); both
are gone on reload. Before each message it rebuilds `query` from its own query state, so
edits made in the query panel reach the model: entities, clusters and relations are taken from
the panel, properties from the last chat query while the entity still has the same name. This
API accepts that shape as it is (there is a test for it). `history` is sent back as received,
so the lock works without extra code. "New search" clears both, which is the reset.

Saved and shared sessions do not include the chat: they store the map, the converted query and
the first sentence, not `history`. A shared link therefore carries no conversation and no lock.

**Errors.** The frontend shows the `message` field of an error body, which is why every error
of this API has one. The chat branch reads `detail` as well, which still works.

**What is logged where.** This API logs no message text. The frontend's route logs every chat
message with the pseudonymous user name to the platform log, and sends the message text to
Matomo for users who allowed analytics (`trackAction("chat", "message", ...)`); server errors
are stored in MongoDB with the message. The NLP API stores every sentence with its result and
the user name in MongoDB (`nlpRequests`); chat turns are stored nowhere in that form, which
matters if real conversations are to be labelled later.

**What the older chat branch lacked** (check each point against the current frontend code):

1. A declined or unanswered first message. The first sentence goes through the search steps:
   the (empty) query is converted, passes `/validate-spot-query`, and the area and search
   steps follow as for a real search. When `refusal` is set or `query.entities` is empty, show
   `reply` and stop there.
2. `refusal.url` is not rendered, so the self-harm reply has no clickable helpline link, and a
   locked chat looks like any other. See "Declined messages" for what to render.
3. Replies are rendered as Markdown with clickable links and images. Switch both off (see
   "Render everything as plain text").
4. Send `X-Spot-User: anonymizeUser(session)` with the key (rate limit per user).
5. On 429, 502 and 504 the typed message is cleared from the input; keep it. A 401 from this
   API (wrong key) is turned into a 500, which is fine.
6. `history` grows without limit and is refused with 422 above 200 messages. Send the latest
   40 or so; the model reads 20.
7. `limitations` is not shown. Optional, since `reply` mentions them.
8. The README of the branch links this API as `dw-innovation/kid2-spot-chat`, a repository
   that does not exist yet.

**Deployment.** The three backend services run from the `docker-compose.yml` of
`dw-innovation/SPOT` (OSM query on port 3000, NLP on 4000, tag search on 5000); this API
is not in that file yet and would be a fourth service. Settings for it: `SPOT_API_KEY` (the
same value as `SPOT_CHAT_API_KEY` in the frontend), `LLM_TURN_TIMEOUT=50`,
`SPOT_CORS_ORIGINS` set to the frontend's address (any value other than `*` closes it, since
no browser calls this API), `SPOT_DEBUG` unset.

## Going public

What the server does on its own, and what the deployment has to add.

| Topic | In the server | To do when deploying |
|---|---|---|
| Who may call it | `SPOT_API_KEY` is checked if set. A key that ships in a browser frontend is readable by every visitor, so it only helps when the frontend's own backend calls this API, which is the case in SPOT. | Set `SPOT_API_KEY`. Set `SPOT_CORS_ORIGINS` to the frontend's origin(s) instead of `*` if browsers call the API directly; in SPOT they do not, so any value other than `*` closes it. The server logs a warning at start while either is open. |
| Request floods | `SPOT_RATE_LIMIT` per user (`X-Spot-User`, believed only together with the key) or else per client address, counted in memory per worker process. | Put a reverse proxy in front for TLS and a second, global limit. If browsers call the API directly, start uvicorn with `--forwarded-allow-ips=<proxy address>`, otherwise every user has the proxy's address and shares one limit. |
| Cost | Message, history and request size are capped; the client cannot pick a model outside `LLM_MODELS`. | Set a spending limit on the LLM account. |
| Forged history | Only `user` and `assistant` turns are accepted. A reply that copies the system prompt word for word is replaced by an off-topic refusal. | Nothing. A client can still invent assistant turns or drop the lock, since the state lives in the client (accepted, see `docs/ROADMAP.md`, decision 3). |
| Internals | `/docs`, `tool_calls` and the details of `/health` are off unless `SPOT_DEBUG=1`. Errors of the LLM provider go to the log, not to the client. | Leave `SPOT_DEBUG` unset in production. |
| Logging | One line per `/chat` call on the logger `spot.api`: outcome (`ok`, a refusal category, `llm_error`, `rate_limited`), reported limits, number of tool calls, lengths and duration. No message text and no client address. | Collect the log; a rise in refusal categories or `rate_limited` is the abuse signal. |
| Privacy | Nothing is stored by this server. | Every message, the chat history and the query are sent to the LLM provider behind `LLM_BASE_URL`, and object names go to the tag search. The privacy notice of the public site has to say so, and it depends on that provider's retention terms. |
