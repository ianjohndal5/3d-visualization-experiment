"""
phase4_reconstruct.py  ─  Phase 4: 3D Reconstruction
──────────────────────────────────────────────────────
Pipeline:
  1. Back-project depth frames into 3-D point clouds using camera intrinsics.
  2. Register consecutive clouds with ICP.
  3. Merge into a single global cloud.
  4. Post-process: voxel downsample + statistical outlier removal.
  5. (Optional) Poisson surface reconstruction.
  6. Save as scene.ply (point cloud) or scene_mesh.obj (mesh).

Usage:
  python phase4_reconstruct.py [--depth-dir depth_samples] [--calib calibration.npz]
                               [--voxel 0.02] [--mesh] [--out scene.ply]
"""

import argparse, sys, numpy as np
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("open3d not installed.  Run:  pip install open3d")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--depth-dir",  default="depth_samples",
                    help="Folder of .npy depth maps from phase3_depth.py --save")
parser.add_argument("--calib",      default="calibration.npz",
                    help="Calibration file from phase2_capture.py calibrate")
parser.add_argument("--voxel",      type=float, default=0.005,
                    help="Voxel down-sample size in metres (default 0.005)")
parser.add_argument("--depth-min",  type=float, default=0.2,
                    help="Minimum depth in metres (default 0.2)")
parser.add_argument("--depth-max",  type=float, default=1.5,
                    help="Maximum depth in metres (default 1.5)")
parser.add_argument("--mesh",       action="store_true",
                    help="Run Poisson surface reconstruction and export mesh")
parser.add_argument("--out",        default="scene.ply",
                    help="Output filename (default scene.ply)")
args = parser.parse_args()

DEPTH_DIR = Path(args.depth_dir)
CALIB     = Path(args.calib)

# ─────────────────────────────────────────────────────────────────────────────
#  Intrinsics
# ─────────────────────────────────────────────────────────────────────────────

def load_intrinsics() -> tuple[float, float, float, float]:
    """Returns (fx, fy, cx, cy) from calibration.npz or safe defaults."""
    if CALIB.exists():
        data = np.load(str(CALIB))
        K    = data["K"]
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        print(f"  [warn]  '{CALIB}' not found - using default intrinsics "
              f"(640×480 webcam approximation).")
        # Generic 640×480 approximation
        return 525.0, 525.0, 319.5, 239.5

# ─────────────────────────────────────────────────────────────────────────────
#  Back-projection: depth map → Open3D point cloud
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_cloud(depth: np.ndarray,
                   fx: float, fy: float,
                   cx: float, cy: float,
                   d_min: float = 0.2,
                   d_max: float = 1.0) -> o3d.geometry.PointCloud:
    # ── [Improvement] Non-linear Inverse Depth ──
    # AI models usually predict disparity/inverse depth. 
    # Linear mapping causes distortions; inverse mapping is physically truer.
    z_raw = depth.astype(np.float64)
    if z_raw.max() > 1.0:
        # Normalize to 0-1
        z_norm = (z_raw - z_raw.min()) / (z_raw.max() - z_raw.min() + 1e-8)
        # Map to metric range using inverse logic: 1/depth
        # High intensity (255) = close, Low intensity (0) = far
        inv_min, inv_max = 1.0/d_min, 1.0/d_max
        z = 1.0 / (inv_max + (1.0 - z_norm) * (inv_min - inv_max))
    else:
        z = z_raw

    h, w = z.shape
    v_idx, u_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Mask invalid pixels + bit of margin to remove edge noise
    mask = (z > d_min) & (z < d_max)
    # Crop borders slightly where AI models are least accurate
    border = 10
    mask[:border, :] = False; mask[-border:, :] = False
    mask[:, :border] = False; mask[:, -border:] = False
    z = z[mask]; u = u_idx[mask]; v = v_idx[mask]

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

    pts = np.column_stack([X, Y, Z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# ─────────────────────────────────────────────────────────────────────────────
#  ICP registration
# ─────────────────────────────────────────────────────────────────────────────

def register_icp(source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud,
                 voxel: float = 0.02) -> np.ndarray:
    """Fast-ICP with point-to-plane.  Returns 4×4 transform."""
    def prep(pcd):
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
        return down

    src_d = prep(source)
    tgt_d = prep(target)

    result = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d,
        max_correspondence_distance=voxel * 5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    )
    print(f"    ICP fitness={result.fitness:.4f}  rmse={result.inlier_rmse:.4f}")
    return result.transformation

# ─────────────────────────────────────────────────────────────────────────────
#  Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def post_process(pcd: o3d.geometry.PointCloud,
                 voxel: float) -> o3d.geometry.PointCloud:
    print(f"  Voxel downsample  (voxel_size={voxel}) …")
    pcd = pcd.voxel_down_sample(voxel)
    print(f"  Statistical outlier removal (relaxed) …")
    # relaxed from 2.0 to 5.0 to keep more points
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)
    return pcd

# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    npy_files = sorted(DEPTH_DIR.glob("*.npy"))
    if not npy_files:
        print(f"No .npy depth maps found in '{DEPTH_DIR}'.\n"
              f"Run:  python phase3_depth.py mono --save\n"
              f"to generate sample depth maps first.")
        sys.exit(1)

    fx, fy, cx, cy = load_intrinsics()
    print(f"\n{'─'*55}")
    print(f"  3D Reconstruction")
    print(f"{'─'*55}")
    print(f"  Intrinsics  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  Depth maps: {len(npy_files)}")
    print(f"  Voxel size: {args.voxel} m\n")

    global_cloud = o3d.geometry.PointCloud()
    cumulative_T = np.eye(4)                  # running transform
    prev_pcd     = None

    for i, npy_path in enumerate(npy_files):
        print(f"  [{i+1}/{len(npy_files)}]  {npy_path.name}")
        depth = np.load(str(npy_path))
        pcd   = depth_to_cloud(depth, fx, fy, cx, cy, args.depth_min, args.depth_max)
        
        # ── [Optimization] Downsample immediately to keep memory usage low ──
        pcd = pcd.voxel_down_sample(args.voxel)

        if prev_pcd is not None and len(pcd.points) > 100:
            # ── [Fix] Estimate normals for PointToPlane ICP ──
            # pcd is already downsampled now
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel*3, max_nn=30))
            
            # prev_pcd is also already downsampled from previous iteration
            if not prev_pcd.has_normals():
                prev_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel*3, max_nn=30))

            result = o3d.pipelines.registration.registration_icp(
                pcd, 
                prev_pcd,
                max_correspondence_distance=args.voxel * 10,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            
            print(f"    ICP fitness={result.fitness:.4f}  rmse={result.inlier_rmse:.4f}")
            
            if result.fitness > 0.5:  # Only add frame if it aligned well
                cumulative_T = cumulative_T @ np.linalg.inv(result.transformation)
                pcd.transform(cumulative_T)
                global_cloud += pcd
                prev_pcd = pcd
            else:
                print(f"    [!] Skipping frame {npy_path.name} - poor alignment (fitness < 0.5)")
        else:
            # First frame or first good frame
            global_cloud += pcd
            prev_pcd = pcd


    if len(global_cloud.points) == 0:
        print("  No points accumulated. Check depth map values.")
        sys.exit(1)

    print(f"\n  Merged cloud:  {len(global_cloud.points):,} points")

    # ── Post-process ──────────────────────────────────────────────────────────
    global_cloud = post_process(global_cloud, args.voxel)
    print(f"  After cleanup: {len(global_cloud.points):,} points")

    # ── Save point cloud ──────────────────────────────────────────────────────
    out_ply = Path(args.out)
    o3d.io.write_point_cloud(str(out_ply), global_cloud)
    print(f"\n  ✓  Point cloud saved → {out_ply}")

    # ── Optional: Poisson mesh ────────────────────────────────────────────────
    if args.mesh:
        print("\n  Running Poisson surface reconstruction …")
        global_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel * 3, max_nn=30))
        global_cloud.orient_normals_consistent_tangent_plane(100)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            global_cloud, depth=9)

        # Remove low-density vertices
        keep = np.array(densities) > np.quantile(np.array(densities), 0.05)
        mesh = mesh.select_by_index(np.where(keep)[0])
        mesh.compute_vertex_normals()

        mesh_path = out_ply.with_name("scene_mesh.obj")
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"  ✓  Mesh saved → {mesh_path}")

    print("\nDone.  Run phase5_visualise.py to view the scene.")

if __name__ == "__main__":
    main()
