"""Builds + executes the project notebooks without needing Jupyter: each code cell is run with exec() in one shared
namespace, stdout / last expression / matplotlib figures are stored as real nbformat-4 outputs.
Re-run:  python tools/build_notebooks.py        (or open the .ipynb files in Jupyter and Run All)
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}
def code(src): return {"cell_type": "code", "metadata": {}, "source": src.strip("\n").splitlines(True), "outputs": [], "execution_count": None}


def run_cell(src: str, ns: dict, count: int) -> tuple[list, bool]:
    outs, ok = [], True
    buf = io.StringIO()
    value = None
    try:
        tree = ast.parse(src)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                value = eval(compile(last, "<cell>", "eval"), ns)
    except Exception:  # noqa: BLE001
        ok = False
        outs.append({"output_type": "error", "ename": "Error", "evalue": "cell failed", "traceback": traceback.format_exc().splitlines()})
    if buf.getvalue():
        outs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
    plt.close("all")
    if value is not None:
        data = {"text/plain": repr(value).splitlines(True)}
        if hasattr(value, "_repr_html_"):
            data["text/html"] = value._repr_html_().splitlines(True)
        outs.append({"output_type": "execute_result", "metadata": {}, "execution_count": count, "data": data})
    return outs, ok


def build(name: str, cells: list[dict]) -> bool:
    ns: dict = {"__name__": "__nb__"}
    all_ok, n = True, 0
    for c in cells:
        if c["cell_type"] == "code":
            n += 1
            c["outputs"], ok = run_cell("".join(c["source"]), ns, n)
            c["execution_count"] = n
            all_ok &= ok
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks" / f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    errs = sum(1 for c in cells if c["cell_type"] == "code" for o in c["outputs"] if o["output_type"] == "error")
    print(f"{name}: {n} code cells, {errs} errors")
    return all_ok


if __name__ == "__main__":
    from tools.notebook_cells import ALL
    ok = True
    for nm, cells in ALL.items():
        ok &= build(nm, cells)
    sys.exit(0 if ok else 1)
