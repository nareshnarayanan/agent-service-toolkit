from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode
from agent import agent_registry
from agent.tools import web_search, arxiv_search


class AgentState(MessagesState):
    is_last_step: IsLastStep

# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
# models = {
#     "llama3.1-70b-cerebras": ChatOpenAI(
#         model="llama3.1-70b",
#         api_key=os.getenv("CEREBRAS_API_KEY"),
#         base_url="https://api.cerebras.ai/v1",
#         # stream=True
#         ),
#     "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", streaming=True),
#     "llama-3.1-70b": ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5),
# }

tools = [web_search, arxiv_search]
current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web for information.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """

def wrap_model(model: BaseChatModel):
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig):
    m = agent_registry.get_llm()
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")

# Always run "model" after "tools"
agent.add_edge("tools", "model")

# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", END: END})

duckduckgo_agent = agent.compile()