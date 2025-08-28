from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, TypedDict
import asyncio

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI

# ----------------------
# Context Definition
# ----------------------
class Context(TypedDict):
    """Parameters that can be configured when invoking the graph"""
    openai_api_key: str  # LLM API key
    system_prompt: str   # Optional system instruction


# ----------------------
# State Definition
# ----------------------
@dataclass
class State:
    """Input state for the agent"""
    user_message: str
    bot_response: str = ""  # Filled after LLM generates response


# ----------------------
# Node 1: Validate Input
# ----------------------
async def validate_input(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    if not state.user_message.strip():
        return {"error": "Message cannot be empty"}
    return {"validated": True}


# ----------------------
# Node 2: Generate LLM Response
# ----------------------
async def generate_response(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-4",
        openai_api_key=runtime.context["openai_api_key"]
    )

    system_prompt = runtime.context.get("system_prompt", "You are a helpful chatbot.")
    prompt = [
{"role": "system", "content": system_prompt},
        {"role": "user", "content": state.user_message}
    ]

    response = await llm.acall(messages=prompt)
    state.bot_response = response.content
    return {"bot_response": state.bot_response}


# ----------------------
# Node 3: Post-process / Optional
# ----------------------
async def post_process(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    # Example: add metadata, log, or clean response
    processed_response = state.bot_response.strip()
    return {"bot_response": processed_response}


# ----------------------
# Define the Graph
# ----------------------
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(validate_input, name="validate_input")
    .add_node(generate_response, name="generate_response")
    .add_node(post_process, name="post_process")
    .add_edge("__start__", "validate_input")
    .add_edge("validate_input", "generate_response")
    .add_edge("generate_response", "post_process")
    .compile(name="Chatbot Graph")
)
