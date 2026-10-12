"""KID2 SPOT - chat demo for building map search queries.

Proof of concept: describe what you are looking for, and the assistant turns it
into a structured query (area, entities, relations) that you can refine in chat.
The query uses everyday object names; mapping them to OpenStreetMap tags is a
separate step that is not part of this demo.

Run locally:  python app.py   (configure the LLM via .env, see spot_chat/config.py)
"""

from __future__ import annotations

import copy
import traceback

import gradio as gr
import yaml

from spot_chat.agent import ErrorEvent, ReplyEvent, ToolEvent, run_turn
from spot_chat.config import load_settings, make_client
from spot_chat.descriptors import get_descriptors
from spot_chat.query import QueryError, empty_query, from_yaml, to_yaml

settings = load_settings()
client = make_client(settings)
if not settings.api_key:
    print("Warning: LLM_API_KEY is not set.")
print(f"Model: {settings.model} | endpoint: {settings.base_url or 'OpenAI default'}")
descriptors = get_descriptors()
if descriptors:
    print(f"SPOT tables: {len(descriptors.entities)} object descriptors, "
          f"{len(descriptors.properties)} property descriptors, {len(descriptors.colours)} colours")

EXAMPLES = [
    "show me all bars next to rivers in cologne",
    "Find a sally port and 3 Pakistani eateries all within 500 meters in Comuna Râfov, Prahova County, Romania.",
    "I'm trying to find a halal butcher in a mall that has elevators and steel walls",
    "Is there an Esso gas station in Pennsylvania, US",
    "Find me an indoor climbing wall that is higher than 3 meters, within 300 meters from a bus station in Groningen.",
    "Please search for a McDonalds in Hidalgo, Mexico.",
    "find me a park with a fountain in stuttgart close to a thalia",
    "Looking for 8 flower shops in Cologne that are at least 50 meters apart",
]

PLACEHOLDER = """**Describe a place you are looking for.**

For example: *"a playground next to a café with outdoor seating in Vienna"*.
Then refine it: *"make it within 200 m"*, *"add a bus stop nearby"*, *"search in Graz instead"*.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def tool_message(event: ToolEvent) -> dict:
    """A collapsible chat bubble that shows one tool call."""
    result = event.result
    args = yaml.safe_dump(result.arguments, sort_keys=False, allow_unicode=True).strip()
    body = result.message
    if result.warnings or result.notes:
        body += "\n\n" + "\n".join(f"- {w}" for w in result.warnings + result.notes)
    body += f"\n\n```yaml\n{args}\n```"
    title = f"🛠️ {result.name}" if result.ok else f"⚠️ {result.name} was rejected, retrying"
    return {"role": "assistant", "content": body, "metadata": {"title": title, "status": "done"}}


def note(text: str) -> dict:
    return {"role": "assistant", "content": f"*{text}*"}


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #

def respond(message, chat, conversation, query, undo_stack, model):
    """Handle a chat message. Yields after every tool call so the YAML updates live."""
    message = (message or "").strip()
    if not message:
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return

    before = copy.deepcopy(query)
    chat = chat + [{"role": "user", "content": message}]
    yield "", chat, gr.skip(), gr.skip(), gr.skip(), gr.skip()

    try:
        for event in run_turn(client, settings, conversation, message, query, model=model):
            query = event.query
            if isinstance(event, ToolEvent):
                chat = chat + [tool_message(event)]
            elif isinstance(event, ReplyEvent):
                chat = chat + [{"role": "assistant", "content": event.text}]
                conversation = conversation + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": event.text},
                ]
            elif isinstance(event, ErrorEvent):
                chat = chat + [note(event.text)]
            yield gr.skip(), chat, to_yaml(query), conversation, query, gr.skip()
    except Exception as exc:  # keep the demo alive and show what went wrong
        traceback.print_exc()
        chat = chat + [note(f"Something went wrong: {exc}")]
        yield gr.skip(), chat, to_yaml(query), conversation, query, gr.skip()

    if query != before:
        undo_stack = undo_stack + [before]
    yield gr.skip(), chat, to_yaml(query), conversation, query, undo_stack


def undo(chat, query, undo_stack):
    if not undo_stack:
        gr.Info("Nothing to undo.")
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    query = undo_stack[-1]
    chat = chat + [note("Undid the last change.")]
    return chat, to_yaml(query), query, undo_stack[:-1]


def apply_manual_edit(yaml_text, chat, query, undo_stack):
    try:
        edited = from_yaml(yaml_text)
    except QueryError as exc:
        gr.Warning(f"Your edit was not applied: {exc}")
        return gr.skip(), to_yaml(query), gr.skip(), gr.skip()
    if edited == query:
        gr.Info("The YAML is unchanged.")
        return gr.skip(), to_yaml(query), gr.skip(), gr.skip()
    chat = chat + [note("Applied your manual edits to the query.")]
    return chat, to_yaml(edited), edited, undo_stack + [query]


def reset():
    query = empty_query()
    return [], to_yaml(query), [], query, [], ""


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

with gr.Blocks(title="SPOT query builder") as demo:
    conversation_state = gr.State([])   # text-only history sent to the LLM
    query_state = gr.State(empty_query())
    undo_state = gr.State([])           # previous queries, newest last

    gr.Markdown(
        "## SPOT query builder\n"
        "Describe what you are looking for on the map. The assistant builds the search query "
        "on the right and changes it when you ask."
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=560, placeholder=PLACEHOLDER, feedback_options=None, label="Chat",
                group_consecutive_messages=False,
            )
            entry = gr.Textbox(
                placeholder="e.g. a bakery close to a tram stop in Freiburg", show_label=False,
                submit_btn="Send", autofocus=True,
            )
            with gr.Row():
                undo_btn = gr.Button("Undo last change", size="sm")
                clear_btn = gr.Button("Start over", size="sm")
        with gr.Column(scale=2):
            yaml_view = gr.Code(
                value=to_yaml(empty_query()), language="yaml", label="Query", interactive=True,
                lines=22, max_lines=40, buttons=["copy", "download"],
            )
            apply_btn = gr.Button("Apply my YAML edits", size="sm")
            model_picker = gr.Dropdown(
                choices=settings.models, value=settings.model, label="Model",
                visible=len(settings.models) > 1,
            )
    gr.Examples(EXAMPLES, inputs=[entry], label="Examples")

    entry.submit(
        respond,
        inputs=[entry, chatbot, conversation_state, query_state, undo_state, model_picker],
        outputs=[entry, chatbot, yaml_view, conversation_state, query_state, undo_state],
    )
    undo_btn.click(
        undo, inputs=[chatbot, query_state, undo_state],
        outputs=[chatbot, yaml_view, query_state, undo_state],
    )
    apply_btn.click(
        apply_manual_edit, inputs=[yaml_view, chatbot, query_state, undo_state],
        outputs=[chatbot, yaml_view, query_state, undo_state],
    )
    clear_btn.click(
        reset, outputs=[chatbot, yaml_view, conversation_state, query_state, undo_state, entry],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch()
