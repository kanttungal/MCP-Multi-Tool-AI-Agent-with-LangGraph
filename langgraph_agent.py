import asyncio
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

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

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 1. MCP CLIENT
# ============================================================

client = MultiServerMCPClient(
    {
        "calculator": {
            "transport": "stdio",
            "command": r"D:\mcp-calculator-server\venv\Scripts\python.exe",
            "args": [
                r"D:\mcp-calculator-server\server.py"
            ],
        }
    }
)


# ============================================================
# 2. MAIN
# ============================================================

async def main():

    # ========================================================
    # 3. OPENROUTER API KEY
    # ========================================================

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Please configure it in your .env file."
        )


    # ========================================================
    # 4. OPENROUTER LLM
    # ========================================================

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


    # ========================================================
    # 5. LOAD MCP TOOLS
    # ========================================================

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )


    # ========================================================
    # 6. BIND TOOLS TO LLM
    # ========================================================

    llm_with_tools = llm.bind_tools(tools)


    # ========================================================
    # 7. AGENT NODE
    # ========================================================

    async def agent_node(state: MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }


    # ========================================================
    # 8. TOOL NODE
    # ========================================================

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True
    )


    # ========================================================
    # 9. CREATE GRAPH
    # ========================================================

    graph_builder = StateGraph(MessagesState)


    # ========================================================
    # 10. ADD AGENT NODE
    # ========================================================

    graph_builder.add_node(
        "agent",
        agent_node
    )


    # ========================================================
    # 11. ADD TOOL NODE
    # ========================================================

    graph_builder.add_node(
        "tools",
        tool_node
    )


    # ========================================================
    # 12. START → AGENT
    # ========================================================

    graph_builder.add_edge(
        START,
        "agent"
    )


    # ========================================================
    # 13. AGENT → TOOLS OR END
    # ========================================================

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )


    # ========================================================
    # 14. TOOLS → AGENT
    # ========================================================

    graph_builder.add_edge(
        "tools",
        "agent"
    )


    # ========================================================
    # 15. CREATE CHECKPOINTER
    # ========================================================

    checkpointer = InMemorySaver()


    # ========================================================
    # 16. COMPILE GRAPH WITH CHECKPOINTER
    # ========================================================

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )


    # ========================================================
    # 17. THREAD ID
    # ========================================================

    config = {
        "configurable": {
            "thread_id": "user_001"
        }
    }


    # ========================================================
    # 18. CHAT LOOP
    # ========================================================

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:

            print("\nGoodbye!")

            break


        # ====================================================
        # RUN GRAPH
        # ====================================================

        result = await graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            },
            config=config
        )


        # ====================================================
        # GET FINAL MESSAGE
        # ====================================================

        final_message = result["messages"][-1]


        print("\nAgent:")

        print(final_message.content)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())