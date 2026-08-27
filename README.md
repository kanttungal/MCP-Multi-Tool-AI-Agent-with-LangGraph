# MCP Multi-Tool AI Agent with LangGraph

A multi-tool AI Agent built using **Model Context Protocol (MCP)**, **LangChain**, **LangGraph**, and **OpenRouter**.

The project demonstrates how an LLM can dynamically select and execute tools exposed by a custom MCP server.

---

## 🚀 Project Overview

This project combines:

- Custom MCP Server
- MCP Client
- OpenRouter LLM
- LangChain
- LangGraph
- Multiple tools
- Tool calling
- Conditional agent routing
- Error handling
- Conversation state
- Checkpointing
- Thread-based conversations

The main goal is to understand how modern AI Agents interact with external tools through MCP.

---

## 🏗️ Architecture

```text
                    User
                      |
                      v
              +---------------+
              |  LangGraph    |
              |     Agent     |
              +-------+-------+
                      |
                      v
              +---------------+
              |     LLM       |
              |   OpenRouter  |
              +-------+-------+
                      |
                Tool Selection
                      |
                      v
              +---------------+
              |   ToolNode    |
              +-------+-------+
                      |
                      v
              +---------------+
              |   MCP Client  |
              +-------+-------+
                      |
                      v
        +---------------------------+
        |       MCP Server          |
        |                           |
        |  add                      |
        |  subtract                 |
        |  multiply                 |
        |  divide                   |
        |  get_weather              |
        |  word_count               |
        +---------------------------+
                      |
                      v
                  Tool Result
                      |
                      v
                    LLM
                      |
                      v
                Final Answer