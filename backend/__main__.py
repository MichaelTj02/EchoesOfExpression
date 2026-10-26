"""Headless run: python -m backend IMAGE [IMAGE ...] [--steps N]

All images are analyzed together and painted onto the canvas at the same time.
"""
import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path

from . import EchoesEngine, LLMConnectionError, MissingAPIKeyError
from .config import MAX_UPLOADS_PER_BATCH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turn handwriting images into a shared canvas drawing without the GUI."
    )
    parser.add_argument(
        "images", nargs="+", metavar="image",
        help=f"Path to a handwriting image (up to {MAX_UPLOADS_PER_BATCH}, drawn at the same time)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Cap on drawing loop steps (default: draw until the paintings are finished)",
    )
    args = parser.parse_args()

    # Windows consoles default to a legacy codepage that can't print e.g. Japanese or Arabic text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = EchoesEngine()
    try:
        results = engine.process_many(
            args.images,
            on_progress=lambda stage, msg: print(f"[{stage}] {msg}"),
            draw_steps=args.steps,
        )
    except (MissingAPIKeyError, LLMConnectionError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for index, (path, result) in enumerate(zip(args.images, results), start=1):
        print(f"\n=== [{index}/{len(results)}] {Path(path).name} ===")
        if isinstance(result, Exception):
            print(f"Error: {result}")
            continue
        for f in fields(result):
            print(f"{f.name}: {getattr(result, f.name)}")

    return 1 if all(isinstance(result, Exception) for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
