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
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

st.set_page_config(
    page_title="MCP AI Agent",
    page_icon="🤖",
    layout="centered"
)


def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    return api_key


def create_llm():
    return ChatOpenAI(
        model="openrouter/free",
        api_key=get_api_key(),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=500
    )


async def run_agent(user_input, thread_id, checkpointer):

    client = MultiServerMCPClient({
        "calculator": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["server.py"]
        }
    })

    tools = await client.get_tools()

    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=True)
    )

    graph_builder.add_edge(START, "agent")

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    graph_builder.add_edge("tools", "agent")

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    return await graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )


# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()


# UI
st.title("🤖 MCP + LangGraph AI Agent")
st.caption("Calculator • Weather • Word Count")


with st.sidebar:
    st.header("⚙️ Chat Settings")

    st.write("Thread ID:")
    st.code(st.session_state.thread_id)

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header("🛠️ Available Tools")
    st.write("➕ Calculator")
    st.write("🌤️ Weather")
    st.write("📝 Word Count")


# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("Ask something...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                result = asyncio.run(
                    run_agent(
                        user_input,
                        st.session_state.thread_id,
                        st.session_state.checkpointer
                    )
                )

                response = result["messages"][-1].content

                if not isinstance(response, str):
                    response = str(response)

                st.write(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:

                status_code = getattr(e, "status_code", None)

                if status_code == 402:
                    st.error("OpenRouter credits are insufficient.")

                elif status_code == 401:
                    st.error("OpenRouter API key is invalid or missing.")

                elif status_code == 429:
                    st.error("OpenRouter rate limit reached.")

                elif status_code and status_code >= 500:
                    st.error("LLM provider server error. Please try again.")

                else:
                    st.error(f"{type(e).__name__}: {e}")

