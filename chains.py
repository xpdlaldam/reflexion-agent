import argparse
import datetime
import html
import os
import tempfile
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # holds all the history of agents
from schemas import AnswerQuestion
from groq import Groq, BadRequestError
from langchain_groq import ChatGroq
from langchain_core.tools import tool


def find_tool_calling_model(candidates=None, verbose=True):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    available = [m.id for m in client.models.list().data]

    # Priority order based on your available models — most likely to support tools
    priority = candidates or [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
        "groq/compound",
        "groq/compound-mini",
    ]

    # Only test models actually in your account
    to_test = [m for m in priority if m in available]

    # Simple probe tool
    @tool
    def probe(x: int) -> int:
        """Test tool."""
        return x

    for model_id in to_test:
        try:
            llm = ChatGroq(model=model_id)
            llm.bind_tools([probe]).invoke("test")
            if verbose:
                print(f"✓ Tool calling works: {model_id}")
            return model_id
        except BadRequestError as e:
            if "tool" in str(e).lower():
                if verbose:
                    print(f"✗ No tool calling: {model_id}")
                continue
            raise  # re-raise if it's a different error

    raise ValueError(f"None of these models support tool calling: {to_test}")

parser = JsonOutputToolsParser(return_id=True) # returns in json/dictionary
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion]) # takes the response from the LLM and search for the function calling location and parse it and transform it into answer-question object => take the ansewr from the LLM and create an answer-question object which we can easily work with

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"), # prompt engineering technique to reuse this promt template => will also be used by the revisor node => will need this for the revisor agent i.e. to revisit the chat history and critique
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
) # to populate already known placeholders. When we invokce this template we want to plug in here the current date. We use the lambda function to output the date with the ISO format

## prepare prompts before sending to LLM
# take the actor prompt template and populate the first instruction field
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

def build_chain(model_id):
    llm = ChatGroq(model=model_id)
    return first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion], tool_choice="AnswerQuestion"
    ) | parser_pydantic


def format_terminal(result, model_id):
    queries = "\n".join(
        f"  {index}. {query}" for index, query in enumerate(result.search_queries, 1)
    )
    return f"""\
Reflexion Agent
Model: {model_id}

ANSWER
{result.answer}

REFLECTION
Missing:
{result.reflection.missing}

Superfluous:
{result.reflection.superfluous}

RECOMMENDED SEARCHES
{queries or "  None"}
"""


def format_html(result, model_id):
    escape = html.escape
    queries = "".join(f"<li>{escape(query)}</li>" for query in result.search_queries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reflexion Agent</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 850px; margin: 0 auto; padding: 2rem; line-height: 1.6; }}
    header {{ border-bottom: 1px solid #8885; margin-bottom: 2rem; }}
    .meta {{ color: #888; }}
    section {{ margin: 2rem 0; }}
    .card {{ border: 1px solid #8885; border-radius: 10px; padding: 1rem 1.25rem; }}
    pre {{ white-space: pre-wrap; font: inherit; margin: 0; }}
  </style>
</head>
<body>
  <header><h1>Reflexion Agent</h1><div class="meta">Model: {escape(model_id)}</div></header>
  <section><h2>Answer</h2><div class="card"><pre>{escape(result.answer)}</pre></div></section>
  <section><h2>Reflection</h2><div class="card">
    <h3>Missing</h3><p>{escape(result.reflection.missing)}</p>
    <h3>Superfluous</h3><p>{escape(result.reflection.superfluous)}</p>
  </div></section>
  <section><h2>Recommended searches</h2><div class="card"><ol>{queries or "<li>None</li>"}</ol></div></section>
</body>
</html>
"""


def write_html(result, model_id, output_path=None, open_browser=False):
    path = Path(output_path) if output_path else Path(tempfile.gettempdir()) / "reflexion-agent.html"
    path.write_text(format_html(result, model_id), encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Ask the Reflexion Agent a question.")
    cli.add_argument("--html", metavar="PATH", help="Write the result to a standalone HTML file.")
    cli.add_argument("--open", action="store_true", help="Open the HTML result in the default browser.")
    args = cli.parse_args()

    human_message = HumanMessage(
        content="Give me advice on how to collect Pokemon cards TCG that will hopefully increase in value over time"
        "Also my budget is max $1,000"
    )
    model_id = find_tool_calling_model()
    print(f"Using model: {model_id}\n")
    chain = build_chain(model_id)
    res = chain.invoke(input={"messages": [human_message]})
    print(format_terminal(res, model_id))

    if args.html or args.open:
        path = write_html(res, model_id, args.html, args.open)
        print(f"HTML report: {path}")
