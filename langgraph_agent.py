import asyncio
import os
from langchain_core.messages import HumanMessage
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

def create_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
            "Please configure it your .env file."
        )

    return ChatOpenAI(
        model = "openai/gpt-4o-mini",
        api_key = api_key,
        base_url = "https://openrouter.ai/api/v1",
    )

# ============================================================
# 2. MAIN
# ============================================================

async def main():

    llm = create_llm()

    # ========================================================
    # 3. LOAD MCP TOOLS
    # ========================================================

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )

    # ========================================================
    # 4. BIND MCP TOOLS TO LLM
    # ========================================================

    llm_with_tools = llm.bind_tools(tools)

    # ========================================================
    # 5. CREATE AGENT NODE
    # ========================================================

    async def agent_node(state:MessagesState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages":[response]
        }

    # ========================================================
    # 6. CREATE TOOL NODE
    # ========================================================

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True
    )

    # ========================================================
    # 7.CHECKPOINTER
    # ========================================================

    checkpointer = InMemorySaver()

    # ========================================================
    # 8. CREATE LANGGRAPH
    # ========================================================

    graph_builder = StateGraph(MessagesState)

    # ========================================================
    # 9. ADD NODES
    # ========================================================

    graph_builder.add_node(
        "agent",
        agent_node
    )

    graph_builder.add_node(
        "tools",
        tool_node
    )

    # ========================================================
    # 10. START → AGENT
    # ========================================================

    graph_builder.add_edge(
        START,
        "agent"
    )

    # ========================================================
    # 11. AGENT → TOOLS OR END
    # ========================================================

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools":"tools",
            END: END
        }
    )

    # ========================================================
    # 12. TOOLS → AGENT
    # ========================================================

    graph_builder.add_edge(
        "tools",
        "agent"
    )

    # ========================================================
    # 13. COMPILE GRAPH
    # ========================================================

    graph = graph_builder.compile(
        checkpointer = checkpointer
    )

    # ========================================================
    # 14. MULTI-TURN CHAT
    # ========================================================

    thread_id = "user_001"

    print("\n===================================")
    print("   MCP + LangGraph AI Agent")
    print("   Type 'exit' to quit")
    print("===================================")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

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

        final_message = result["messages"][-1]

        print("\nAgent:")
        print(final_message.content)

    # ========================================================
    # 15. PRINT FINAL RESPONSE
    # ========================================================

        final_message = result["messages"][-1]

        print("\nAgent:")
        print(final_message.content)

if __name__ == "__main__":
    asyncio.run(main())