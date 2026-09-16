import asyncio
import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END,
)
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="MCP AI Agent",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """
    Get OpenRouter API key.

    Local:
        .env

    Streamlit Cloud:
        Streamlit Secrets
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to your .env locally or Streamlit Secrets on Cloud."
        )

    return api_key


# ============================================================
# MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["server.py"],
        }
    }
)


# ============================================================
# LLM
# ============================================================

def create_llm():

    api_key = get_api_key()

    return ChatOpenAI(
        model="openrouter/free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=500,
    )


# ============================================================
# CREATE LANGGRAPH
# ============================================================

async def create_graph():

    llm = create_llm()

    # Get MCP tools
    tools = await client.get_tools()

    # Bind MCP tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------------
    # AGENT NODE
    # --------------------------------------------------------

    async def agent_node(state: MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    # --------------------------------------------------------
    # TOOL NODE
    # --------------------------------------------------------

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True
    )

    # --------------------------------------------------------
    # CHECKPOINTER
    # --------------------------------------------------------

    checkpointer = InMemorySaver()

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(
        "agent",
        agent_node
    )

    graph_builder.add_node(
        "tools",
        tool_node
    )

    # START → AGENT
    graph_builder.add_edge(
        START,
        "agent"
    )

    # AGENT → TOOLS or END
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    # TOOLS → AGENT
    graph_builder.add_edge(
        "tools",
        "agent"
    )

    # Compile graph
    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    return graph


# ============================================================
# RUN GRAPH
# ============================================================

async def run_graph(
    graph,
    user_input,
    thread_id
):

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return result


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(
        uuid.uuid4()
    )


if "messages" not in st.session_state:

    st.session_state.messages = []


if "graph" not in st.session_state:

    try:

        st.session_state.graph = asyncio.run(
            create_graph()
        )

    except Exception as e:

        st.error(
            f"Failed to initialize AI Agent: {e}"
        )

        st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🤖 MCP + LangGraph AI Agent")

st.caption(
    "Calculator • Weather • Word Count"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Chat Settings")

    st.write("Thread ID:")

    st.code(
        st.session_state.thread_id,
        language="text"
    )

    if st.button(
        "🆕 New Chat",
        use_container_width=True
    ):

        st.session_state.thread_id = str(
            uuid.uuid4()
        )

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.header("🛠️ Available Tools")

    st.write("➕ Calculator")
    st.write("🌤️ Weather")
    st.write("📝 Word Count")


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask something..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_input:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.write(user_input)

    # --------------------------------------------------------
    # RUN AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = asyncio.run(
                    run_graph(
                        st.session_state.graph,
                        user_input,
                        st.session_state.thread_id
                    )
                )

                # ------------------------------------------------
                # GET FINAL MESSAGE
                # ------------------------------------------------

                final_message = result["messages"][-1]

                assistant_response = (
                    final_message.content
                )

                # Some models can return structured content.
                if not isinstance(
                    assistant_response,
                    str
                ):

                    assistant_response = str(
                        assistant_response
                    )

                # ------------------------------------------------
                # DISPLAY FINAL RESPONSE ONLY
                # ------------------------------------------------

                st.write(
                    assistant_response
                )

                # ------------------------------------------------
                # SAVE ASSISTANT RESPONSE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_response
                    }
                )

            # ----------------------------------------------------
            # API / GENERAL ERROR HANDLING
            # ----------------------------------------------------

            except Exception as e:

                error_type = type(e).__name__

                # API status code if available
                status_code = getattr(
                    e,
                    "status_code",
                    None
                )

                if status_code == 402:

                    st.error(
                        "OpenRouter credits are insufficient "
                        "for this request. Please use an "
                        "available free model or add credits."
                    )

                elif status_code == 401:

                    st.error(
                        "OpenRouter API key is invalid "
                        "or missing."
                    )

                elif status_code == 429:

                    st.error(
                        "OpenRouter rate limit reached. "
                        "Please try again later."
                    )

                elif status_code and status_code >= 500:

                    st.error(
                        "The LLM provider returned a server error. "
                        "Please try again."
                    )

                else:

                    st.error(
                        f"{error_type}: {str(e)}"
                    )