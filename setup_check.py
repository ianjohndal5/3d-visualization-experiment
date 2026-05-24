"""
setup_check.py  ─  Phase 1: Verify all dependencies are installed correctly.
Run with:  python setup_check.py
"""

import sys
import importlib
from typing import Callable

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"

def check(label: str, import_name: str, version_fn: Callable | None = None):
    try:
        mod = importlib.import_module(import_name)
        if version_fn:
            version = version_fn(mod)
        else:
            version = getattr(mod, "__version__", "unknown")
        print(f"  {GREEN}✓{RESET}  {label:<30} {CYAN}{version}{RESET}")
        return True
    except Exception as exc:
        print(f"  {RED}✗{RESET}  {label:<30} {RED}FAILED – {exc}{RESET}")
        return False

print(f"\n{BOLD}{'─'*55}")
print(f"  3D Scanner – Dependency Check")
print(f"{'─'*55}{RESET}")
print(f"  Python  {sys.version.split()[0]}  ({sys.executable})\n")

results = []

# ── Core ──────────────────────────────────────────────────
print(f"{BOLD}  Core Libraries{RESET}")
results.append(check("OpenCV",        "cv2",         lambda m: m.__version__))
results.append(check("NumPy",         "numpy",       lambda m: m.__version__))
results.append(check("Pillow",        "PIL",         lambda m: m.__version__))
results.append(check("Matplotlib",    "matplotlib",  lambda m: m.__version__))

# ── 3D ────────────────────────────────────────────────────
print(f"\n{BOLD}  3D Reconstruction{RESET}")
results.append(check("Open3D",        "open3d",      lambda m: m.__version__))
results.append(check("SciPy",         "scipy",       lambda m: m.__version__))
results.append(check("Trimesh",       "trimesh",     lambda m: m.__version__))

# ── Deep Learning ─────────────────────────────────────────
print(f"\n{BOLD}  Deep Learning{RESET}")
results.append(check("PyTorch",       "torch",       lambda m: m.__version__))
results.append(check("Torchvision",   "torchvision", lambda m: m.__version__))
results.append(check("Transformers",  "transformers",lambda m: m.__version__))
results.append(check("HuggingFace Hub","huggingface_hub", lambda m: m.__version__))

# ── Visualisation ─────────────────────────────────────────
print(f"\n{BOLD}  Visualisation{RESET}")
results.append(check("PyVista",       "pyvista",     lambda m: m.__version__))

# ── Utilities ─────────────────────────────────────────────
print(f"\n{BOLD}  Utilities{RESET}")
results.append(check("TQDM",          "tqdm",        lambda m: m.__version__))

# ── Hardware summary ──────────────────────────────────────
print(f"\n{BOLD}  Hardware / Acceleration{RESET}")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        print(f"  {GREEN}✓{RESET}  CUDA available – {torch.cuda.get_device_name(0)}")
        print(f"       CUDA version: {torch.version.cuda}")
    else:
        print(f"  {YELLOW}!{RESET}  CUDA not available – running on CPU")
        print(f"       Depth estimation will be slower but functional.")
except Exception as e:
    print(f"  {RED}✗{RESET}  Could not query torch CUDA status: {e}")

# ── Optional: Intel RealSense ─────────────────────────────
print(f"\n{BOLD}  Optional: Intel RealSense{RESET}")
try:
    import pyrealsense2 as rs
    print(f"  {GREEN}✓{RESET}  pyrealsense2 found  (version: {rs.__version__})")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) > 0:
        print(f"  {GREEN}✓{RESET}  RealSense camera detected: {devices[0].get_info(rs.camera_info.name)}")
    else:
        print(f"  {YELLOW}!{RESET}  No RealSense camera connected.")
except ImportError:
    print(f"  {YELLOW}!{RESET}  pyrealsense2 not installed (optional – needed only for RealSense).")
except Exception as e:
    print(f"  {RED}✗{RESET}  RealSense error: {e}")

# ── Summary ───────────────────────────────────────────────
passed = sum(results)
total  = len(results)
print(f"\n{'─'*55}")
if passed == total:
    print(f"  {GREEN}{BOLD}All {total} checks passed!  Your environment is ready.{RESET}")
else:
    print(f"  {YELLOW}{BOLD}{passed}/{total} checks passed.  Fix the red items above, then re-run.{RESET}")
print(f"{'─'*55}\n")
