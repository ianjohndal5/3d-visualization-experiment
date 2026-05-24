"""
utils.py  ─  Shared helpers for all phase scripts
───────────────────────────────────────────────────
Call `silence_qt()` BEFORE the main script body runs to suppress
the Qt/Wayland/QFontDatabase warnings on Gnome Wayland systems.
"""

import os, sys


# ─────────────────────────────────────────────────────────────────────────────
#  Qt / OpenCV warning suppressor
# ─────────────────────────────────────────────────────────────────────────────

def silence_qt() -> None:
    """
    Suppress all Qt/Wayland/QFontDatabase C-level stderr noise.

    IMPORTANT: Qt's display backend (font loading, Wayland plugin, locale) is
    initialised **lazily on the first window creation**, NOT at import time.
    Therefore we must:
      1. Redirect fd-2 (C-level stderr) to /dev/null.
      2. Import cv2.
      3. Force full Qt initialisation by creating + destroying a 1×1 window.
      4. Restore fd-2.
    After this call every subsequent cv2.imshow() runs in an already-
    initialised Qt environment and emits no further warnings.
    """
    # ── Point Qt at real system fonts so QFontDatabase doesn't complain ──
    # NOTE: Set OPENCV_LOG_LEVEL before import, but QT_QPA_FONTDIR must be
    # set AFTER import cv2 — cv2 resets it at import time to its bundle path.
    os.environ["OPENCV_LOG_LEVEL"]  = "ERROR"
    os.environ["QT_LOGGING_RULES"]  = "*.debug=false;qt.qpa.*=false;qt.text.*=false"

    # ── Flush Python stderr buffers before touching the raw fd ───────────
    sys.stderr.flush()

    # ── Redirect C-level stderr (fd 2) → /dev/null ───────────────────────
    saved_fd = os.dup(2)
    devnull  = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        import cv2  # noqa: F401

        # Set font dir HERE, after cv2 import (cv2 resets it during import).
        for font_dir in (
            "/usr/share/fonts/gnu-free",
            "/usr/share/fonts/noto",
            "/usr/share/fonts/liberation",
            "/usr/share/fonts/TTF",
            "/usr/share/fonts",
        ):
            if os.path.isdir(font_dir):
                os.environ["QT_QPA_FONTDIR"] = font_dir
                break

        # Force the full lazy Qt display-backend init while fd-2 is silent.
        cv2.namedWindow("__qt_init__", cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)                   # pump Qt event loop → fonts + Wayland init
        cv2.destroyWindow("__qt_init__")
        cv2.waitKey(1)                   # second pump ensures destroy is flushed

    finally:
        # ── Restore C-level stderr ────────────────────────────────────────
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour map helper  (shared by phase3 & phase4)
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_colormap(depth_hw: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Normalise a float32 (H, W) depth array → uint8 BGR heatmap."""
    import cv2, numpy as np
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("inferno")
    d = depth_hw.astype(np.float32).copy()
    valid = d[d > 0]
    if valid.size:
        d = np.clip(d, valid.min(), valid.max())
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    rgba = (cmap(d) * 255).astype(np.uint8)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
