from __future__ import annotations

import argparse
from pathlib import Path

from tic_agent import create_workflow, load_config


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the TIC research workflow")
    parser.add_argument("question", nargs="?", default="水冷板 行业 TIC 机会", help="研究问题")
    parser.add_argument("--stream", action="store_true", help="流式打印中间步骤")
    args = parser.parse_args()

    config = load_config(Path(".env"))
    workflow = create_workflow(config)

    if args.stream:
        for event in workflow.run(input=args.question, stream=True):  # type: ignore
            if hasattr(event, "content") and event.content:
                print(event.content, end="", flush=True)
        print()
        return

    result = workflow.run(input=args.question)
    print(result.content)


if __name__ == "__main__":
    run_cli()
