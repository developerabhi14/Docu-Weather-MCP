# Agentic Knowledge Assistant with LangGraph and MCP

This project demonstrates an **agentic AI assistant** that dynamically decides which tool to use for a given user query.  
It integrates **MCP servers** for tool execution with a **LangGraph workflow** for intelligent routing and response formatting.

---

## Features

- Multi-tool agent powered by LangGraph
- MCP servers:
  - **Weather API** – fetches current weather for a city (stub/demo implementation)
  - **RAG Tool** – semantic search over local documents using FAISS and HuggingFace embeddings
- Router node: uses LLM to decide which tool to invoke based on user query
- Formatter node: generates a clean, concise response for the user
- Modular, extensible design for adding new tools

---

## Project Structure

