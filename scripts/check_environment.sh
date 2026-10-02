#!/usr/bin/env bash
# Print package versions, GPU and data paths at the start of every job (a missing package
# is printed as MISSING; only a failing `import sparseshower.common` stops the job).
set -euo pipefail
echo "--- environment ---"
python - <<'PY'
import importlib
import platform
import sys

print("python      :", sys.version.split()[0], platform.platform())
for name in ("numpy", "yaml", "pyarrow", "scipy", "matplotlib", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:<12}:", getattr(mod, "__version__", "?"))
    except ImportError:
        print(f"{name:<12}: MISSING")
try:
    import torch

    print("cuda        :", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"gpu memory  : {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")
        cap = torch.cuda.get_device_capability(0)
        print(f"capability  : {cap[0]}.{cap[1]}")
        # is_bf16_supported() is True on pre-Ampere cards too (emulated)
        print("bf16 native :", cap[0] >= 8)
except ImportError:
    pass
PY
echo "--- paths ---"
python - "$@" <<'PY'
import os
import sys

from sparseshower.common import config

cfg = os.environ.get("SPARSE_CONFIG", "configs/v2.yaml")
try:
    c = config(cfg)
except Exception as exc:  # noqa: BLE001
    print(f"config {cfg}: {type(exc).__name__}: {exc}")
    sys.exit(0)
for key, value in c["paths"].items():
    print(f"{key:<10}: {value}  exists={os.path.isdir(value)}")
PY
