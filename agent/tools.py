from langchain_core.tools import tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from typing import Annotated

web_search = DuckDuckGoSearchRun(name="WebSearch")

# Kinda busted since it doesn't return links
arxiv_search = ArxivQueryRun(name="ArxivSearch")

tavily_search = TavilySearchResults(name="TavilySearch")

# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )