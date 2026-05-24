"""
phase2_capture.py  ─  Phase 2: Camera input & calibration
─────────────────────────────────────────────────────────
Modes (pass as first CLI arg):
  capture     - live feed; press 's' save frame, 'q' quit       (default)
  calibrate   - detect chessboard corners → calibration.npz
  stream      - live undistorted feed + ORB keypoints overlay

Usage:
  python phase2_capture.py [capture|calibrate|stream]
"""

import os, sys

# ── MUST happen before ANY cv2 import ────────────────────────────────────────
from utils import silence_qt
silence_qt()
# ─────────────────────────────────────────────────────────────────────────────

import cv2, numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
CAPTURE_W      = 640          # keep low to avoid OOM on 3.5 GB RAM systems
CAPTURE_H      = 480
SAVE_DIR       = Path("frames")
CALIBRATION    = Path("calibration.npz")
CHESS_ROWS     = 9            # inner corners (rows)
CHESS_COLS     = 6            # inner corners (cols)
SQUARE_MM      = 25.0         # physical square size in mm
MAX_CALIB_IMGS = 40

# ── Helpers ───────────────────────────────────────────────────────────────────

def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Open a camera, auto-scanning all /dev/video* devices if the requested
    index fails.  Many USB cameras register two V4L2 nodes (image + metadata);
    this finds the one that actually delivers frames.
    """
    import glob, re

    def _try(idx: int, backend=cv2.CAP_V4L2) -> cv2.VideoCapture | None:
        """Return a working VideoCapture or None."""
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        # Verify we can actually read a frame (not just open the fd)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
        return None

    # 1. Try the requested index first
    cap = _try(index) or _try(index, 0)
    if cap:
        print(f"  Camera opened at index {index}")
        return cap

    # 2. Auto-scan /dev/video0 … /dev/video9
    video_nodes = sorted(glob.glob("/dev/video?"))
    indices = [int(m.group()) for p in video_nodes if (m := re.search(r"\d+", p))]
    print(f"  Index {index} failed - scanning {video_nodes} …")
    for idx in indices:
        if idx == index:
            continue
        cap = _try(idx) or _try(idx, 0)
        if cap:
            print(f"  Camera opened at index {idx}  (/dev/video{idx})")
            return cap

    raise RuntimeError(
        f"No working camera found (tried indices {[index] + indices}).\n\n"
        "Fix options:\n"
        "  1. Add yourself to the video group then re-login:\n"
        "       sudo usermod -aG video $USER\n"
        "  2. Temporarily grant access without logout:\n"
        "       sudo chmod a+rw /dev/video0\n"
        "  3. Verify the webcam is physically connected:\n"
        "       ls /dev/video*"
    )


def draw_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 150), 1, cv2.LINE_AA)
    return out


def make_orb() -> cv2.ORB:
    return cv2.ORB_create(nfeatures=800)       # fewer features → less RAM


def draw_keypoints(frame: np.ndarray, orb: cv2.ORB) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kps  = orb.detect(gray, None)
    vis  = cv2.drawKeypoints(frame, kps, None,
                             color=(0, 220, 255),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return vis, len(kps)

# ── Mode: capture ─────────────────────────────────────────────────────────────

def mode_capture():
    SAVE_DIR.mkdir(exist_ok=True)
    cap = open_camera()
    idx = 0
    print(f"[capture]  Resolution: {CAPTURE_W}×{CAPTURE_H}")
    print("           Press 's' to save a frame, 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed - check camera connection.")
            break

        out = draw_overlay(frame, f"LIVE  |  saved:{idx}  |  s=save  q=quit")
        cv2.imshow("3D Scanner - Capture", out)
        key = cv2.waitKey(30) & 0xFF       # 30 ms → ~33 fps max, reduces CPU/RAM pressure

        if key == ord('s'):
            path = SAVE_DIR / f"frame_{idx:04d}.png"
            cv2.imwrite(str(path), frame)
            print(f"  Saved → {path}")
            idx += 1
        elif key in (ord('q'), 27):        # q or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture done. {idx} frame(s) in '{SAVE_DIR}/'.")

# ── Mode: calibrate ───────────────────────────────────────────────────────────

def mode_calibrate():
    imgs = sorted(SAVE_DIR.glob("frame_*.png"))
    if not imgs:
        sys.exit(f"No frames in '{SAVE_DIR}/'. Run capture mode first.")

    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2) * SQUARE_MM

    obj_pts, img_pts = [], []
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    for path in imgs[:MAX_CALIB_IMGS]:
        img  = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (CHESS_COLS, CHESS_ROWS), flags)
        symbol = "✓" if ok else "✗"
        print(f"  {symbol}  {path.name}", end="")
        if ok:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            vis = cv2.drawChessboardCorners(
                img.copy(), (CHESS_COLS, CHESS_ROWS), corners2, ok)
            cv2.imshow("Calibration - Chessboard Corners", vis)
            cv2.waitKey(250)
            used += 1
            print(f"  [{used} good]")
        else:
            print("  (corners not found)")

    cv2.destroyAllWindows()

    if used < 6:
        sys.exit(f"\nNeed ≥6 good frames; only {used} found. "
                 "Capture more chessboard images (try different angles/distances).")

    h, w = cv2.imread(str(imgs[0])).shape[:2]
    rms, K, dist, *_ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)

    np.savez(str(CALIBRATION), K=K, dist=dist, rms=float(rms), w=w, h=h)
    print(f"\nCalibration complete  (RMS reprojection error: {rms:.4f} px)")
    print(f"  Intrinsic matrix K:\n{K}")
    print(f"  Distortion coefficients: {dist.ravel()}")
    print(f"  Saved → {CALIBRATION}")

# ── Mode: stream ──────────────────────────────────────────────────────────────

def mode_stream():
    if not CALIBRATION.exists():
        sys.exit(f"'{CALIBRATION}' not found. Run calibrate mode first.")

    data   = np.load(str(CALIBRATION))
    K, dist = data["K"], data["dist"]
    h, w    = int(data["h"]), int(data["w"])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, dist, None, newK, (w, h), cv2.CV_32FC1)

    orb = make_orb()
    cap = open_camera()
    print("[stream]  Undistorted feed + ORB keypoints.  q/Esc = quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        undist       = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        vis, kp_count = draw_keypoints(undist, orb)
        vis = draw_overlay(vis,
            f"UNDISTORTED + ORB  |  keypoints:{kp_count}  |  q=quit")
        cv2.imshow("3D Scanner - Stream", vis)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── Entry ─────────────────────────────────────────────────────────────────────

MODES = {"capture": mode_capture, "calibrate": mode_calibrate, "stream": mode_stream}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    fn = MODES.get(mode)
    if fn is None:
        sys.exit(f"Unknown mode '{mode}'. Use: {' | '.join(MODES)}")
    fn()
