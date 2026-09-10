import asyncio
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START,
    END,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

import os,sys
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                "server.py"
            ],
        }
    }
)


# ============================================================
# LLM
# ============================================================

def create_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens = 1000,
    )


# ============================================================
# CREATE LANGGRAPH
# ============================================================

async def create_graph():

    llm = create_llm()

    tools = await client.get_tools()

    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True
    )

    checkpointer = InMemorySaver()

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(
        "agent",
        agent_node
    )

    graph_builder.add_node(
        "tools",
        tool_node
    )

    graph_builder.add_edge(
        START,
        "agent"
    )

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    graph_builder.add_edge(
        "tools",
        "agent"
    )

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    return graph


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="MCP AI Agent",
    page_icon="🤖"
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 MCP + LangGraph AI Agent")

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

with st.sidebar:

    st.header("🛠️ Available Tools")

    st.write("➕ Calculator")
    st.write("🌤️ Weather")
    st.write("📝 Word Count")

st.caption(
    "Calculator • Weather • Word Count"
)


# ============================================================
# CREATE GRAPH
# ============================================================

if "graph" not in st.session_state:

    st.session_state.graph = asyncio.run(
        create_graph()
    )


# ============================================================
# THREAD ID
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())

# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


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

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    result = asyncio.run(
        st.session_state.graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            },
            config={
                "configurable": {
                    "thread_id":
                    st.session_state.thread_id
                }
            }
        )
    )


    # ============================================================
# SHOW TOOL ACTIVITY
# ============================================================

    for message in result["messages"]:


        if hasattr(message, "tool_calls") and message.tool_calls:


           for tool_call in message.tool_calls:


            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            with st.status(
                f"🔧 Using tool: {tool_name}",
                expanded=False
            ):

                st.write("Arguments:")

                st.json(tool_args)

    

    final_message = result["messages"][-1]

    assistant_response = final_message.content

    st.session_state.messages.append(
       {
           "role": "assistant",
           "content": assistant_response
       }
)

    with st.chat_message("assistant"):
      st.write(assistant_response)
