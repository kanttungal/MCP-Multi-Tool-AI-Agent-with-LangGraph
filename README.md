# 🤖 MCP + LangGraph AI Agent

A multi-tool AI Agent built using **LangGraph, LangChain, Model Context Protocol (MCP), OpenRouter, and Streamlit**.

The agent can understand a user's request, decide which tool is required, execute the appropriate tool through an MCP server, and return the final response.

---

# ✨ Features

- 🤖 AI Agent using LangGraph
- 🔌 Model Context Protocol (MCP) integration
- 🧮 Multi-tool support
- 🧠 LLM-based tool selection
- 🔄 Agent → Tool → Agent workflow
- 🛡️ Tool error handling
- 💾 Conversation state using LangGraph Checkpointer
- 🆔 Thread-based conversation handling
- 💬 Interactive Streamlit Chat UI
- 🔐 OpenRouter LLM integration
- ⚡ Asynchronous MCP tool execution
- 🔗 LangChain + LangGraph + MCP integration

---

# 🛠️ Available Tools

The MCP server provides the following tools:

## 🧮 Calculator

The agent can perform basic mathematical operations.

### Tools

```text
add
subtract
multiply
divide


                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LangGraph Agent   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    OpenRouter LLM   │
                    └──────────┬──────────┘
                               │
                        Tool Selection
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ToolNode       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     MCP Server      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Calculator          Weather         Word Count


START
  │
  ▼
AGENT
  │
  ▼
Tool Required?
  │
  ├────────────── No ──────────────► END
  │
  ▼
TOOLS
  │
  ▼
AGENT
  │
  ▼
END

mcp-calculator-server/
│
├── server.py
├── client.py
├── langchain_client.py
├── langgraph_agent.py
├── streamlit_app.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env


| Technology        | Purpose                                |
| ----------------- | -------------------------------------- |
| **Python**        | Core programming language              |
| **LangChain**     | LLM and tool integration               |
| **LangGraph**     | Agent workflow and state management    |
| **MCP**           | Standardized tool/server communication |
| **OpenRouter**    | LLM API provider                       |
| **ChatOpenAI**    | OpenAI-compatible LLM interface        |
| **Streamlit**     | Interactive web UI                     |
| **Pydantic**      | Data validation                        |
| **asyncio**       | Asynchronous execution                 |
| **python-dotenv** | Environment variable management        |


streamlit run streamlit_app.py

┌──────────────────────────────────────────────┐
│ 🤖 MCP + LangGraph AI Agent                 │
│ Calculator • Weather • Word Count           │
│                                              │
│ You: Calculate 25 * 8                        │
│                                              │
│ Agent:                                       │
│ 25 × 8 = 200                                 │
│                                              │
│ Ask something...                             │
└──────────────────────────────────────────────┘

🚀 How to Run
1. Create virtual environment
python -m venv venv
2. Activate environment

Windows:

venv\Scripts\activate
3. Install dependencies

pip install -r requirements.txt

4. Configure API key

Create:

.env

and add:

OPENROUTER_API_KEY=your_api_key_here
5. Start Streamlit
streamlit run streamlit_app.py