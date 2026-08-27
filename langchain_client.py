import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
async def main():

    # =====================================================
    # 1. LLM
    # =====================================================

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )


    # =====================================================
    # 2. MCP CLIENT
    # =====================================================

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


    # =====================================================
    # 3. GET MCP TOOLS
    # =====================================================

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(
            f"- {tool.name}: {tool.description}"
        )


    # =====================================================
    # 4. BIND MCP TOOLS TO LLM
    # =====================================================

    llm_with_tools = llm.bind_tools(tools)


    # =====================================================
    # 5. ASK LLM
    # =====================================================

    response = await llm_with_tools.ainvoke(
        "Calculate 25 multiplied by 8"
    )


    print("\nLLM Response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())