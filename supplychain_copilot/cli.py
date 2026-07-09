"""Terminal chat with the agent.

    python -m supplychain_copilot.cli                 # interactive REPL
    python -m supplychain_copilot.cli "one question"  # single-shot
"""

import sys

from .agent.graph import ask, build_agent


def main() -> None:
    agent = build_agent()
    if len(sys.argv) > 1:
        print(ask(agent, " ".join(sys.argv[1:])))
        return
    print("Supply Chain Copilot — ask about inventory, forecasts, shipments, contracts.")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        print("\ncopilot>", ask(agent, q), "\n")


if __name__ == "__main__":
    main()
