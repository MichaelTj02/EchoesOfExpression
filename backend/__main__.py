"""Headless run: python -m backend IMAGE [--steps N]"""
import argparse
import logging
import sys
from dataclasses import fields

from . import EchoesEngine, MissingAPIKeyError, NoTextFoundError


def main() -> int:
    parser = argparse.ArgumentParser(description="Turn a handwriting image into a canvas drawing without the GUI.")
    parser.add_argument("image", help="Path to a handwriting image")
    parser.add_argument("--steps", type=int, default=None, help="Drawing loop steps (default: random 3000-4500)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = EchoesEngine()
    try:
        result = engine.process(
            args.image,
            on_progress=lambda stage, msg: print(f"[{stage}] {msg}"),
            draw_steps=args.steps,
        )
    except (MissingAPIKeyError, NoTextFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    for f in fields(result):
        print(f"{f.name}: {getattr(result, f.name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
