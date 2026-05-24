# 🔭 3D Environmental Scanner

A Python-based 3D scanner pipeline covering capture → depth estimation → reconstruction → visualisation.

## Requirements
- Python **3.12** (via pyenv)
- No CUDA required — CPU inference supported

---

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify all dependencies
python setup_check.py
```

---

## Phase 1 — Environment Setup

| File | Purpose |
|------|---------|
| `requirements.txt` | All dependencies |
| `setup_check.py` | Import check + hardware summary |

```bash
# Install deps (already done)
.venv/bin/pip install -r requirements.txt

# Verify
.venv/bin/python setup_check.py
```

### Camera Driver Notes

**USB Webcam (OpenCV default)**
```bash
# Test camera is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```
On Linux, ensure your user is in the `video` group:
```bash
sudo usermod -aG video $USER   # then log out and back in
```

**Intel RealSense (optional)**
```bash
# Install librealsense (Arch)
sudo pacman -S librealsense

# Install Python bindings
.venv/bin/pip install pyrealsense2
```

---

## Phase 2 — Camera Capture & Calibration

```bash
# 1. Live capture — press 's' to save frames, 'q' to quit
python phase2_capture.py capture

# 2. Calibrate using saved chessboard frames
#    Print a 9×6 chessboard, photograph it from ~20 angles
python phase2_capture.py calibrate

# 3. Live undistorted stream + ORB keypoints
python phase2_capture.py stream
```

Output: `calibration.npz` (intrinsic matrix K + distortion coefficients)

---

## Phase 3 — Depth Estimation

### Option A — Monocular (default, single webcam)
```bash
# Downloads DepthAnything v2 Small from HuggingFace on first run (~100 MB)
python phase3_depth.py mono --save
```

### Option B — Stereo (two cameras)
```bash
python phase3_depth.py stereo --save
```

### Option C — Intel RealSense
```bash
python phase3_depth.py realsense --save
```

`--save` writes `.npy` depth arrays + PNG heatmaps to `depth_samples/`.

---

## Phase 4 — 3D Reconstruction

```bash
# Point cloud only
python phase4_reconstruct.py --depth-dir depth_samples --calib calibration.npz

# With Poisson mesh output
python phase4_reconstruct.py --mesh --out scene.ply
```

Outputs: `scene.ply` (point cloud), `scene_mesh.obj` (mesh, if `--mesh` used)

---

## Phase 5 — Visualisation & Export

```bash
# Open3D viewer (r=reset, p=screenshot, q=quit)
python phase5_visualise.py view --file scene.ply

# PyVista viewer (polished, height-coloured)
python phase5_visualise.py pyvista --file scene.ply

# Export to .glb (Three.js/web) and/or .obj (Unity)
python phase5_visualise.py export --file scene.ply --glb --obj

# Save scan session (point cloud + poses)
python phase5_visualise.py save_session --file scene.ply

# Resume a saved session
python phase5_visualise.py resume --session session.npz
```

---

## Full Pipeline (end to end)

```bash
source .venv/bin/activate

# 1. Capture ~30 chessboard frames
python phase2_capture.py capture

# 2. Calibrate
python phase2_capture.py calibrate

# 3. Collect depth maps (scan your environment)
python phase3_depth.py mono --save

# 4. Reconstruct 3D scene
python phase4_reconstruct.py --mesh

# 5. View and export
python phase5_visualise.py view
python phase5_visualise.py export --glb --obj
```

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| opencv-python | 4.13.0 | Capture, calibration, stereo |
| open3d | 0.19.0 | Point cloud, ICP, visualisation |
| torch (CPU) | 2.12.0 | DepthAnything v2 inference |
| transformers | 5.9.0 | HuggingFace model pipeline |
| pyvista | 0.48.4 | Polished 3D viewer |
| trimesh | 4.12.2 | GLB export for web |

> **CUDA:** Install `torch` + `torchvision` from the CUDA index when a GPU is available:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
