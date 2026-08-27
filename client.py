import asyncio
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

load_dotenv()


# ============================================================
# 1. OPENROUTER CLIENT
# ============================================================

llm = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# 2. MCP SERVER CONFIGURATION
# ============================================================

server_params = StdioServerParameters(
    command=r"D:\mcp-calculator-server\venv\Scripts\python.exe",
    args=[
        r"D:\mcp-calculator-server\server.py"
    ],
)


# ============================================================
# 3. MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # Connect to MCP Server
    # --------------------------------------------------------

    async with stdio_client(
        server_params
    ) as (read_stream, write_stream):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            # =================================================
            # STEP 1: INITIALIZE MCP CONNECTION
            # =================================================

            await session.initialize()

            print("\nConnected to MCP Server.")


            # =================================================
            # STEP 2: DISCOVER MCP TOOLS
            # =================================================

            tools_result = await session.list_tools()

            print("\nAvailable MCP Tools:")

            for tool in tools_result.tools:

                print(
                    f"- {tool.name}: "
                    f"{tool.description}"
                )


            # =================================================
            # STEP 3: CONVERT MCP TOOLS → LLM TOOLS
            # =================================================

            llm_tools = []

            for tool in tools_result.tools:

                llm_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )


            # =================================================
            # STEP 4: USER INPUT
            # =================================================

            user_input = input(
                "\nAsk something: "
            )


            # =================================================
            # STEP 5: FIRST LLM CALL
            # =================================================

            messages = [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]

            response = llm.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages,
                tools=llm_tools,
            )


            assistant_message = response.choices[0].message


            # =================================================
            # STEP 6: CHECK WHETHER LLM REQUESTED A TOOL
            # =================================================

            if assistant_message.tool_calls:

                print("\nLLM selected tool(s):")

                # Add assistant's tool-call message
                messages.append(
                    assistant_message
                )


                # =================================================
                # STEP 7: EXECUTE MCP TOOLS
                # =================================================

                for tool_call in assistant_message.tool_calls:

                    tool_name = tool_call.function.name

                    arguments = json.loads(
                        tool_call.function.arguments
                    )


                    print(
                        f"\nTool: {tool_name}"
                    )

                    print(
                        f"Arguments: {arguments}"
                    )


                    # Call MCP tool
                    result = await session.call_tool(
                        tool_name,
                        arguments=arguments,
                    )


                    print(
                        f"MCP Result: {result}"
                    )


                    # =================================================
                    # STEP 8: SEND MCP RESULT TO LLM
                    # =================================================

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )


                # =================================================
                # STEP 9: FINAL LLM RESPONSE
                # =================================================

                final_response = llm.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                )


                final_answer = (
                    final_response
                    .choices[0]
                    .message
                    .content
                )


                print("\nFinal Answer:")
                print(final_answer)


            # =================================================
            # STEP 10: NO TOOL REQUIRED
            # =================================================

            else:

                print("\nFinal Answer:")
                print(
                    assistant_message.content
                )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())