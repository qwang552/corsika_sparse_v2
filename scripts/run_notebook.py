#!/usr/bin/env python
"""Execute a notebook headless (e.g. inside a condor job) - no Jupyter needed.

Runs the code cells in order in one namespace, exactly like "Run All", and
writes an executed copy of the notebook with every cell's printed output and
every figure embedded, so the result can be opened in Jupyter / VS Code later
without re-running anything.

The first code cell is the parameter cell.  `--set NAME=VALUE` replaces the
line `NAME = ...` there before it runs (the value is a Python literal), so one
notebook serves both the GPU and the CPU job:

    python scripts/run_notebook.py notebooks/viz.ipynb \
        --set DEVICE=cpu --set N_RECON=3 --threads 8

An existing executed notebook is never overwritten (an explicit --out that
exists is refused; the default name gets a timestamp instead).

Stops at the first failing cell (later cells depend on earlier ones); the
executed notebook is still written, with the traceback in that cell.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path


class Tee(io.TextIOBase):
    """Write to the job's stdout (so `tail -f` works) and keep a copy."""

    def __init__(self, stream):
        self.stream, self.buf = stream, io.StringIO()

    def write(self, s):
        """Write to the stream and keep a copy."""
        self.stream.write(s)
        self.stream.flush()
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()


def as_literal(value):
    """'3' -> 3, 'False' -> False, 'cpu' -> 'cpu' (bare words become strings, so
    condor arguments never need nested quotes)."""
    try:
        ast.literal_eval(value)
        return value
    except (ValueError, SyntaxError):
        return repr(value)


def patch_parameters(src, settings):
    """Replace `NAME = ...` lines of the parameter cell with the --set values."""
    for name, value in settings.items():
        value = as_literal(value)
        pattern = re.compile(rf'^{re.escape(name)}\s*=.*$', re.M)
        if not pattern.search(src):
            raise SystemExit(f'--set {name}: no line "{name} = ..." in the parameter cell')
        src = pattern.sub(f'{name} = {value}   # set by run_notebook.py', src, count=1)
    return src


def main(argv=None):
    """Run every code cell in one namespace and write the executed notebook with outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('notebook')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='override a parameter in the first code cell (Python literal)')
    ap.add_argument('--out', help='executed notebook path (default: <notebook>_run.ipynb next to it; '
                                  'a timestamp is appended when that exists)')
    ap.add_argument('--threads', type=int, help='CPU threads for torch / BLAS')
    args = ap.parse_args(argv)

    if args.threads:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ[var] = str(args.threads)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass

    nb_path = Path(args.notebook).resolve()
    out_path = Path(args.out) if args.out else nb_path.with_name(nb_path.stem + '_run.ipynb')
    if out_path.exists():                            # never overwrite an executed notebook
        if args.out:
            raise SystemExit(f'{out_path} exists; pass another --out')
        out_path = out_path.with_name(f'{out_path.stem}_{time.strftime("%Y%m%d-%H%M%S")}.ipynb')
    nb = json.loads(nb_path.read_text())
    settings = {}
    for item in args.set:
        name, _, value = item.partition('=')
        settings[name.strip()] = value.strip()

    figures = []

    def show(*_a, **_k):
        """Replacement for plt.show: embed the open figures as PNG in the current cell's output."""
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
            figures.append(base64.b64encode(buf.getvalue()).decode())
        plt.close('all')

    plt.show = show
    ns = {'__name__': '__main__'}
    first_code, failed, count = True, False, 0
    t_all = time.time()
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        cell['outputs'], cell['execution_count'] = [], None
        if failed:
            continue
        src = ''.join(cell['source'])
        if first_code:
            src = patch_parameters(src, settings)
            cell['source'] = src.splitlines(True)
            first_code = False
        title = next((l.strip() for l in src.splitlines() if l.strip()), '')[:70]
        print(f'\n===== cell {count}: {title}', flush=True)
        tee, figures[:] = Tee(sys.stdout), []
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(tee):
                exec(compile(src, f'<cell {count}>', 'exec'), ns)
                show()                               # figures created without plt.show()
        except BaseException as exc:                 # noqa: BLE001 - report and stop
            failed = True
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            cell['outputs'].append(dict(output_type='error', ename=type(exc).__name__, evalue=str(exc),
                                        traceback=tb.splitlines()))
        cell['execution_count'] = count
        text = tee.buf.getvalue()
        if text:
            cell['outputs'].insert(0, dict(output_type='stream', name='stdout', text=text.splitlines(True)))
        for png in figures:
            cell['outputs'].append(dict(output_type='display_data', metadata={},
                                        data={'image/png': png, 'text/plain': ['<Figure>']}))
        print(f'----- cell {count} {"FAILED" if failed else "done"} in {time.time() - t0:.0f} s', flush=True)

    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f'\nexecuted notebook -> {out_path}  ({time.time() - t_all:.0f} s total)', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
