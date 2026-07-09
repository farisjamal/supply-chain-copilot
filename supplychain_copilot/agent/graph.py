"""Agent assembly — LangGraph ReAct agent with tools and memory.

The agent is a LangGraph state machine: the model node reasons and either
answers or emits tool calls; the tool node executes them and loops back.
That loop is the "planning & reasoning" core of the system.

Memory: a LangGraph checkpointer persists the message state per thread_id,
so follow-up questions ("and what about its supplier?") keep full context.
"""

from langgraph.checkpoint.memory import InMemorySaver

try:
    # langchain >= 1.0
    from langchain.agents import create_agent

    _NEW_API = True
except ImportError:  # older stacks
    from langgraph.prebuilt import create_react_agent as create_agent

    _NEW_API = False

from ..config import get_llm
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


def build_agent(llm=None, checkpointer=None):
    """Build the compiled agent graph. Pass a custom llm/checkpointer for tests."""
    llm = llm or get_llm()
    checkpointer = checkpointer or InMemorySaver()
    if _NEW_API:
        return create_agent(
            llm, TOOLS, system_prompt=SYSTEM_PROMPT, checkpointer=checkpointer
        )
    return create_agent(llm, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


def ask(agent, question: str, thread_id: str = "default") -> str:
    """Send one user message; return the agent's final text answer."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content
