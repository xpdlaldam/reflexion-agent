from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool # converts a python function into a tool that can be used by the LLM
from langgraph.prebuilt import ToolNode # a node in langgraph we can invoke which will look the state of the messages key
from langchain_tavily import TavilySearch

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)