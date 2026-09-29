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

Interactive docs are at `http://localhost:8000/docs`.

| Variable | Meaning |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | the language model endpoint |
| `SPOT_TAG_SEARCH_URL` | SPOT's tag search, default `https://tags.findthatspot.io/api/search` |
| `SPOT_API_KEY` | if set, every request needs the header `X-API-Key: <value>` |
| `SPOT_CORS_ORIGINS` | comma-separated origins for browser calls, default `*` |

## POST /chat

One turn of the conversation.

```jsonc
{
  "message": "show me all bars next to rivers in cologne",  // required
  "query": null,        // the query from the previous response; null or omitted starts fresh
  "query_yaml": null,   // alternative to "query": the same thing as a YAML string
  "history": [],        // previous turns, oldest first: {"role": "user"|"assistant", "content": "..."}
  "model": null         // optional, overrides the configured model
}
```

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
  "tool_calls": [        // what the model did, useful for a debug view
    {"name": "new_query", "ok": true, "message": "Started a new query with 2 entities and 1 relations.",
     "arguments": { ... }, "warnings": [], "notes": []}
  ]
}
```

Show `reply` in the chat, and draw the map from `query`. Update the map when `changed` is `true`.

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

`{"status": "ok", "model": ..., "descriptors_loaded": 8606, ...}`. Contains no secrets, so it is
safe for monitoring.

## Status codes

| Code | Meaning |
|---|---|
| 200 | fine |
| 400 | empty message, or the query you sent is invalid (`detail` says what) |
| 401 | `SPOT_API_KEY` is set and the `X-API-Key` header is missing or wrong |
| 502 | the language model could not be reached (`detail` has the reason) |

## Notes

- A turn takes a few seconds, since the model may call several tools. Allow at least 60 seconds
  of client timeout and show a typing indicator.
- `query.entities[].id` is stable inside one query, and relations refer to those ids. After a
  deletion, entities are renumbered, so always redraw from the returned query.
- Cluster entities have `type: "cluster"` with `minPoints` and `maxDistance`.
- Relative spatial terms ("next to") are converted into their distance; only the distance is in
  the query. Set `SPOT_INCLUDE_SPATIAL_TERM=1` to also get the wording.
- The same YAML goes into SPOT's tag search and OSM query steps unchanged.
