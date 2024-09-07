import asyncio
from contextlib import asynccontextmanager
import json
import os
import logging, coloredlogs
import traceback
from typing import AsyncGenerator, Dict, Any, Tuple
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.graph import CompiledGraph
from langsmith import Client as LangsmithClient
from agent.agents import chart_generator, research_assistant, duckduckgo_agent

from agent.agent_registry import AgentRegistry
from schema import ChatMessage, Feedback, UserInput, StreamInput, AgentList, AgentInfo

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format the log messages
    handlers=[
        logging.StreamHandler()  # Output to the terminal
    ]
)
coloredlogs.install(level='INFO')

logger = logging.getLogger(__name__)  # Create a logger for your application

class TokenQueueStreamingHandler(AsyncCallbackHandler):
    """LangChain callback handler for streaming LLM tokens to an asyncio queue."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            await self.queue.put(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Construct agent with Sqlite checkpointer
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        registry = AgentRegistry(load_direrctory="agent/agents")
        for agent in registry.get_all():
            agent.checkpointer = saver
        app.state.registry = registry
        yield
    # context manager will clean up the AsyncSqliteSaver on exit

app = FastAPI(lifespan=lifespan)

# @app.middleware("http")
async def logging_middleware(request: Request, call_next):
    logger.info("Received request: %s", str(request.body))
    response = await call_next(request)
    logger.info("Sent response: %s", str(response.__dict__))
    return response

@app.middleware("http")
async def check_auth_header(request: Request, call_next):
    if auth_secret := os.getenv("AUTH_SECRET"):
        auth_header = request.headers.get('Authorization') 
        if not auth_header or not auth_header.startswith("Bearer "):
            return Response(status_code=401, content="Missing or invalid token")
        if auth_header[7:] != auth_secret:
            return Response(status_code=401, content="Invalid token")
    return await call_next(request)

def _parse_input(user_input: UserInput) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    input_message = ChatMessage(type="human", content=user_input.message)
    kwargs = dict(
        input={"messages": [input_message.to_langchain()]},
        config=RunnableConfig(
            configurable={"thread_id": thread_id},
            run_id=run_id,
        ),
    )
    return kwargs, run_id

@app.post("/invoke")
async def invoke(user_input: UserInput) -> ChatMessage:
    """
    Invoke the agent with user input to retrieve a final response.
    
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    kwargs, run_id = _parse_input(user_input)
    logger.info("Invoking agent[%s] with input: %s, %s", user_input.agent, kwargs, run_id)
    agent: CompiledGraph = app.state.registry.get(user_input.agent)
    try:
        response = await agent.ainvoke(**kwargs)
        output = ChatMessage.from_langchain(response["messages"][-1])
        output.run_id = str(run_id)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def message_generator(user_input: StreamInput) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    logger.info("Invoking agent[%s] with input: %s", user_input.agent, user_input)
    agent: CompiledGraph = app.state.registry.get(user_input.agent)
    kwargs, run_id = _parse_input(user_input)
    logger.info("Starting stream for input: %s, %s", kwargs, run_id)

    # Use an asyncio queue to process both messages and tokens in
    # chronological order, so we can easily yield them to the client.
    output_queue = asyncio.Queue()
    if user_input.stream_tokens:
        kwargs["config"]["callbacks"] = [TokenQueueStreamingHandler(queue=output_queue)]

    # Pass the agent's stream of messages to the queue in a separate task, so
    # we can yield the messages to the client in the main thread.
    async def run_agent_stream():
        try:
            async for s in agent.astream(**kwargs, stream_mode="updates"):
                await output_queue.put(s)
            await output_queue.put(None)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            logger.error("Error streaming messages: %s", e)
            await output_queue.put(f"Error streaming messages: {e}")
            await output_queue.put(None)
    stream_task = asyncio.create_task(run_agent_stream())

    # Process the queue and yield messages over the SSE stream.
    while s := await output_queue.get():
        if isinstance(s, str):
            # str is an LLM token
            wire_msg = f"data: {json.dumps({'type': 'token', 'content': s})}\n\n"
            yield wire_msg
            continue

        # Otherwise, s should be a dict of state updates for each node in the graph.
        # s could have updates for multiple nodes, so check each for messages.
        new_messages = []
        for _, state in s.items():
            new_messages.extend(state["messages"])
        for message in new_messages:
            try:
                chat_message = ChatMessage.from_langchain(message)
                chat_message.run_id = str(run_id)
            except Exception as e:
                wire_msg = f"data: {json.dumps({'type': 'error', 'content': f'Error parsing message: {e}'})}\n\n"
                yield wire_msg
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                continue
            wire_msg = f"data: {json.dumps({'type': 'message', 'content': chat_message.dict()})}\n\n"
            yield wire_msg

    await stream_task
    logger.info("Stream complete")
    yield "data: [DONE]\n\n"

@app.post("/stream")
async def stream_agent(user_input: StreamInput):
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.
    
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")

@app.post("/feedback")
async def feedback(feedback: Feedback):
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return {"status": "success"}


@app.get("/agents")
async def list_agents():
    """
    Return a list of agent names registered with the service.
    """
    agents = app.state.registry.get_all_names()
    logger.debug("Registered Agents: %s", agents)
    return AgentList(agents=agents)

@app.get("/agents/{agent_name}")
async def get_agent_graph(agent_name) -> AgentInfo:
    graph = app.state.registry.get_graph(agent_name)
    logger.info("Agent Graph: %s", graph)
    return AgentInfo(mermaid_graph=graph)