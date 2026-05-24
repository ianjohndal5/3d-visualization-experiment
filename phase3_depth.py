"""
phase3_depth.py  ─  Phase 3: Depth estimation
──────────────────────────────────────────────
Three modes selected automatically:

  mono      - Monocular depth via Depth-Anything v2 (HuggingFace)   [default]
  stereo    - Stereo disparity via cv2.StereoSGBM (needs 2 cameras)
  realsense - Intel RealSense aligned depth + colour

Usage:
  python phase3_depth.py [mono|stereo|realsense] [--save]

Use --save to write sample depth maps to disk (depth_samples/ folder).
"""

import os, sys, time, argparse

# ── Silence Qt/Wayland font noise before cv2 loads ────────────────────────────
from utils import silence_qt
silence_qt()
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np, cv2, matplotlib.pyplot as plt
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("mode",    nargs="?", default="mono",
                    choices=["mono", "stereo", "realsense"],
                    help="Depth estimation mode")
parser.add_argument("--index",  type=int, default=0, help="Camera index")
parser.add_argument("--url",    help="IP Camera URL (e.g. http://192.168.1.15:8080/video)")
parser.add_argument("--save",   action="store_true", help="Save depth map as .npy and .png")
args = parser.parse_args()

CAMERA_INDEX = args.url if args.url else args.index
SAVE_DIR     = Path("depth_samples")

CALIBRATION = Path("calibration.npz")

# ──────────────────────────────────────────────────────────────────────────────
#  Colour map helper
# ──────────────────────────────────────────────────────────────────────────────
_cmap = plt.get_cmap("inferno")

def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Normalize float32 depth → uint8 BGR heatmap."""
    d = depth.copy()
    valid = d[d > 0]
    if valid.size:
        d = np.clip(d, valid.min(), valid.max())
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    rgba = (np.array(_cmap(d)) * 255).astype(np.uint8)   # RGBA
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def save_depth(depth: np.ndarray, tag: str, frame_idx: int):
    if not args.save:
        return
    SAVE_DIR.mkdir(exist_ok=True)
    np.save(str(SAVE_DIR / f"{tag}_{frame_idx:04d}.npy"), depth.astype(np.float32))
    vis = depth_to_colormap(depth)
    cv2.imwrite(str(SAVE_DIR / f"{tag}_{frame_idx:04d}.png"), vis)
    print(f"  Saved depth sample → {SAVE_DIR}/{tag}_{frame_idx:04d}.{{npy,png}}")


# ──────────────────────────────────────────────────────────────────────────────
#  Option A - Monocular (Depth-Anything v2)
# ──────────────────────────────────────────────────────────────────────────────

def run_mono():
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("torch / transformers not installed. Run:  pip install torch transformers")
        sys.exit(1)

    print("Loading Depth-Anything v2 Base (Higher Precision) …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Upgraded to 'Base' model for much better definition of small objects
    pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device=0 if device == "cuda" else -1,
    )
    print(f"  Model loaded on {device.upper()}.  Press 'q' to quit.")

    # ── [New] Process saved frames if they exist ──
    input_dir = Path("frames")
    saved_frames = sorted(input_dir.glob("frame_*.png"))
    
    if saved_frames and not args.url:
        print(f"Found {len(saved_frames)} saved frames in '{input_dir}'. Processing offline...")
        for i, img_path in enumerate(saved_frames):
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            
            # Convert to PIL RGB for the pipeline
            from PIL import Image as PILImage
            rgb_pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result   = pipe(rgb_pil)
            depth_np = np.array(result["depth"], dtype=np.float32)
            
            # ── [AI Improvement] Edge-preserving smoothing ──
            # This helps define the shape and reduces "blobbiness"
            depth_np = cv2.bilateralFilter(depth_np, d=5, sigmaColor=0.1, sigmaSpace=5)
            
            save_depth(depth_np, "mono", i)
            
            heatmap = depth_to_colormap(depth_np)
            cv2.imshow("3D Scanner - Processing Depth", np.hstack([bgr, heatmap]))
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        print("Offline processing complete.")
        cv2.destroyAllWindows()
        return

    # ── Fallback to Live Feed ──
    print(f"Opening camera feed: {CAMERA_INDEX}")
    backend = 0 if isinstance(CAMERA_INDEX, str) else cv2.CAP_V4L2
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera: {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)


    frame_idx = 0
    fps_t = time.time()

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # Convert to PIL RGB for the pipeline
        from PIL import Image as PILImage
        rgb_pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result   = pipe(rgb_pil)
        depth_np = np.array(result["depth"], dtype=np.float32)   # (H, W) relative depth

        save_depth(depth_np, "mono", frame_idx)

        heatmap = depth_to_colormap(depth_np)
        heatmap = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))

        fps = 1.0 / (time.time() - fps_t + 1e-8)
        fps_t = time.time()

        side_by_side = np.hstack([bgr, heatmap])
        cv2.putText(side_by_side, f"Monocular Depth (DepthAnything v2)  FPS:{fps:.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 1)
        cv2.imshow("3D Scanner - Mono Depth", side_by_side)

        if cv2.waitKey(30) & 0xFF == ord('q'):   # 30 ms → ~33 fps cap
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  Option B - Stereo
# ──────────────────────────────────────────────────────────────────────────────

def run_stereo():
    """
    Expects two USB cameras at index 0 and 1.
    Adjust BASELINE_M and focal_px to match your rig.
    """
    BASELINE_M  = 0.06          # metres between lenses
    FOCAL_MM    = 3.6           # approximate focal length in mm
    SENSOR_W_MM = 3.6           # sensor width in mm  (1/4" sensor ≈ 3.6 mm)
    IMG_W_PX    = 640

    focal_px = (FOCAL_MM / SENSOR_W_MM) * IMG_W_PX

    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    for c in (cap0, cap1):
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W_PX)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8*3*5**2, P2=32*3*5**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print("Stereo mode.  Press 'q' to quit.")
    frame_idx = 0

    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()
        if not (r0 and r1):
            print("Cannot read from both cameras.")
            break

        g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        disp  = sgbm.compute(g0, g1).astype(np.float32) / 16.0
        # Mask invalid pixels
        disp[disp <= 0] = np.nan
        # depth (m) = focal_px * BASELINE_M / disparity
        depth = focal_px * BASELINE_M / (disp + 1e-8)
        depth = np.nan_to_num(depth, nan=0.0)

        save_depth(depth, "stereo", frame_idx)

        heatmap = depth_to_colormap(depth)
        out = np.hstack([f0, heatmap])
        cv2.imshow("3D Scanner - Stereo Depth", out)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        frame_idx += 1

    for c in (cap0, cap1):
        c.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  Option C - Intel RealSense
# ──────────────────────────────────────────────────────────────────────────────

def run_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 not installed. Install it via:  pip install pyrealsense2")
        sys.exit(1)

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align   = rs.align(rs.stream.color)

    depth_scale = (profile.get_device()
                   .first_depth_sensor()
                   .get_depth_scale())
    print(f"RealSense depth scale: {depth_scale:.5f} m/unit.  Press 'q' to quit.")

    frame_idx = 0
    try:
        while True:
            frames  = pipe.wait_for_frames()
            aligned = align.process(frames)
            depth_f = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not depth_f or not color_f:
                continue

            depth = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale  # metres
            color = np.asanyarray(color_f.get_data())

            save_depth(depth, "realsense", frame_idx)

            heatmap = depth_to_colormap(depth)
            out = np.hstack([color, heatmap])
            cv2.imshow("3D Scanner - RealSense Depth (m)", out)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            frame_idx += 1
    finally:
        try:
            pipe.stop()
        except:
            pass
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
#  Entry
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    {"mono": run_mono, "stereo": run_stereo, "realsense": run_realsense}[args.mode]()
