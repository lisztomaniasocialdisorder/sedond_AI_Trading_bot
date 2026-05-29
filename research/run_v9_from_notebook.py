#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v9_edge_detector notebook training cell on local machine."
    )
    parser.add_argument(
        "--notebook",
        default=r"C:\Users\brian\trading\notebooks\v9_edge_detector.ipynb",
        help="Path to v9 notebook.",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Trading base directory (contains features/ and experiments/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    base_dir = Path(args.base_dir).resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    nb = json.loads(notebook_path.read_text(encoding="ascii"))
    if len(nb.get("cells", [])) < 5:
        raise RuntimeError("Unexpected notebook structure: training cell not found.")

    training_code = "".join(nb["cells"][4].get("source", []))
    src_decl = 'BASE_DIR = "/content/drive/MyDrive/trading"'
    dst_decl = f'BASE_DIR = r"{str(base_dir)}"'
    if src_decl in training_code:
        training_code = training_code.replace(src_decl, dst_decl, 1)
    else:
        # If declaration format changes in notebook, fail loudly to avoid silent misuse.
        raise RuntimeError("BASE_DIR declaration not found in training cell.")

    glb: dict[str, object] = {"__name__": "__main__"}
    print(f"[run_v9_from_notebook] notebook={notebook_path}")
    print(f"[run_v9_from_notebook] base_dir={base_dir}")
    exec(training_code, glb, glb)


if __name__ == "__main__":
    main()
