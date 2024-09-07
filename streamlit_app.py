import asyncio
import os
from typing import AsyncGenerator, List
from uuid import uuid4

import streamlit as st
import streamlit_mermaid as stmd

from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatMessage


# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Nirvana"
APP_ICON = "🧰"

# DEBUG INFO
LANGSMITH_THREAD_URL_TEMPLATE = "https://smith.langchain.com/o/7a461e0c-7c23-5c4c-9c5d-e979414ee876/projects/p/d4a2cc98-7bf4-49e9-9097-a65c4d3b9f86/t/{thread_id}"

def get_agent_client():
    agent_url = os.getenv("AGENT_URL", "http://localhost")
    return AgentClient(agent_url)

class SessionManager:
    def init():
        if "sessions" not in st.session_state:
            st.session_state.sessions = {}
            st.session_state.current_session_id = get_script_run_ctx().session_id
            st.session_state.sessions[st.session_state.current_session_id] = {
                "messages": [],
            }

    def set_current_session(session_id):
        st.session_state.current_session_id = session_id
    
    def get_current_session_id():
        return st.session_state.current_session_id

    def get_current_session():
        return st.session_state.sessions[st.session_state.current_session_id]

    def new_session():
        session_id = str(uuid4())
        st.session_state.sessions[session_id] = {
            "messages": [],
        }
        SessionManager.set_current_session(session_id)
        return session_id
    
    def get_all_session_ids():
        return [s for s in st.session_state.sessions.keys()]
    
    def get_session(session_id):
        return st.session_state.sessions[session_id]

async def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={
        },
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    @st.dialog("Current Session")
    def show_debug():
        with st.container():
            st.link_button("Langsmith", LANGSMITH_THREAD_URL_TEMPLATE.format(thread_id=SessionManager.get_current_session_id()))
            st.json(SessionManager.get_current_session(), expanded=False)

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        debug = st.button("Debug", key="debug")
        if debug:
            show_debug()

    agents = await get_agent_client().alist_agents()

    # @st.dialog("Agent Info", width="large")
    def graph_agent(graph):
        with st.sidebar.expander("Agent Graph", expanded=True):
            stmd.st_mermaid(graph)
    selected_agent = st.sidebar.selectbox("Agent", agents["agents"])
    agent_info = await get_agent_client().aget_agent_info(selected_agent)
    mermaid = agent_info["mermaid_graph"]\
        .replace("%%{init: {'flowchart': {'curve': 'linear'}}}%%",\
        "%%{init: {'theme': 'dark', 'flowchart': {'curve': 'basis'}}}%%")\
        .replace("#f2f0ff", "#f2f0f")\
        .replace("#bfb6fc", "#754b4b")
    graph_agent(mermaid)

    session = None
    session_id = None
    # List sessions
    with st.sidebar:
        SessionManager.init()
        session_ids = SessionManager.get_all_session_ids()

        with st.columns([1, 1, 1])[1]:
            new_session = st.button("New +")
            if new_session:
                session_id = SessionManager.new_session()
                st.rerun()

        session_id = SessionManager.get_current_session_id()
        session = SessionManager.get_session(session_id)

        for id in session_ids:
            if "title" not in SessionManager.get_session(id):
                title = "New Session : " + id
            else:
                title = SessionManager.get_session(id)["title"]

            display = title
            if len(display) > 36:
                display = display[:36] + "..."

            hasClicked = st.button(label=display, help=title, disabled=id == session_id, use_container_width=True)
            if hasClicked:
                SessionManager.set_current_session(id)
                session_id = id
                st.rerun()

    messages: List[ChatMessage] = session["messages"]

    if len(messages) == 0:
        WELCOME = "Hello! Choose one of the agents above and ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter():
        for m in messages: yield m
    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if input := st.chat_input():
        messages.append(ChatMessage(type="human", content=input))
        if "title" not in session:
            session["title"] = input
        st.chat_message("human").write(input)
        agent_client = get_agent_client()
        stream = agent_client.astream(
            message=input,
            agent=selected_agent,
            thread_id=session_id,
        )
        await draw_messages(stream, is_new=True)
        st.rerun() # Clear stale containers

    # If messages have been generated, show feedback widget
    # TODO: st.feedback is not working as expected.
    # if len(messages) > 0:
    #     with SessionManager.get_current_session()["last_message"]:
    #         await handle_feedback()


async def draw_messages(
        messages_agen: AsyncGenerator[ChatMessage | str, None],
        is_new=False,
    ):
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.
    
    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    SessionManager.get_current_session()["last_message"] = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    SessionManager.get_current_session()["last_message"] = st.chat_message("ai")
                with SessionManager.get_current_session()["last_message"]:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    SessionManager.get_current_session()["messages"].append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    SessionManager.get_current_session()["last_message"] = st.chat_message("ai")
                
                with SessionManager.get_current_session()["last_message"]:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                    f"""**{tool_call["name"]}**\n\n {tool_call["args"]}""",
                                    state="running" if is_new else "complete",
                                )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if not tool_result.type == "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()
                            
                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                SessionManager.get_current_session()["messages"].append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _: 
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback():
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in SessionManager.get_current_session():
        SessionManager.get_current_session()["last_feedback"] = (None, None)

    latest_run_id = SessionManager.get_current_session()["messages"][-1].run_id
    print("Going to render feedback widget for run ID:", latest_run_id)
    feedback = st.feedback(options="thumbs", key=latest_run_id)
    print("Feedback:", feedback)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback and (latest_run_id, feedback) != SessionManager.get_current_session()["last_feedback"]:
        
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client = get_agent_client()
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs=dict(
                comment="In-line human feedback",
            ),
        )
        SessionManager.get_current_session()["last_feedback"] = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
