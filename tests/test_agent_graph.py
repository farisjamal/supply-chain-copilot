"""Smoke test: the agent graph compiles with tools + memory attached.

Uses ChatOllama as the model object because it can be constructed without
a server or API key — nothing is invoked, we only verify graph assembly.
"""

from langchain_ollama import ChatOllama

from supplychain_copilot.agent.graph import build_agent
from supplychain_copilot.agent.tools import TOOLS


def test_agent_graph_compiles():
    agent = build_agent(llm=ChatOllama(model="llama3.2"))
    graph = agent.get_graph()
    node_names = set(graph.nodes)
    assert any("tool" in n for n in node_names), node_names


def test_all_tools_have_descriptions():
    for t in TOOLS:
        assert t.description, f"{t.name} missing docstring"
