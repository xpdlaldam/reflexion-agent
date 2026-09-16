import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # holds all the history of agents
from schemas import AnswerQuestion
from langchain_groq import ChatGroq

# llm = ChatGroq(model="llama-3.3-70b-versatile") # don't hardcode

from groq import Groq
import os


from groq import Groq, BadRequestError
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import os

def find_tool_calling_model(candidates=None):
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
            print(f"✓ Tool calling works: {model_id}")
            return model_id
        except BadRequestError as e:
            if "tool" in str(e).lower():
                print(f"✗ No tool calling: {model_id}")
                continue
            raise  # re-raise if it's a different error

    raise ValueError(f"None of these models support tool calling: {to_test}")

model_id = find_tool_calling_model()
print(f"Using model: {model_id}")

llm = ChatGroq(model=model_id)

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

# first responder chain
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion" # force the LLM to always use the AnswerQuestion tool => grounding the response to the object we want to receive (cool technique!)
)

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Give me advice on how to collect Pokemon cards TCG that will hopefully increase in value over time"
        "Also my budget is max $1,000"
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)