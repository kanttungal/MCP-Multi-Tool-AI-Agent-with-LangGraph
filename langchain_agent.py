import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

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
    # 3. LOAD MCP TOOLS
    # =====================================================

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(
            f"- {tool.name}: "
            f"{tool.description}"
        )


    # =====================================================
    # 4. CREATE AGENT
    # =====================================================

    system_prompt = """
You are a helpful multi-tool AI assistant.

You have access to calculator, weather,
and text-processing tools through an MCP server.

Rules:

1. Use tools when they are required.
2. You may use multiple tools for one user request.
3. If one tool fails, do not crash.
4. Continue with the remaining independent tasks.
5. Clearly explain tool errors in the final answer.
6. Never invent tool results.
7. For weather questions, use the weather tool.
8. For calculations, use calculator tools.
9. For word counting, use the word_count tool.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


    # =====================================================
    # 5. USER INPUT LOOP
    # =====================================================

    print("\nMCP Multi-Tool Agent")
    print("Type 'exit' to stop.")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break


        # =================================================
        # 6. RUN AGENT
        # =================================================

        try:

            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                }
            )


            # =============================================
            # 7. FINAL ANSWER
            # =============================================

            final_message = result["messages"][-1]

            print("\nAgent:")

            print(final_message.content)


        except Exception as e:

            print("\nAgent Error:")
            print(
                "The agent encountered an unexpected error."
            )

            print(f"Details: {e}")


if __name__ == "__main__":
    asyncio.run(main())