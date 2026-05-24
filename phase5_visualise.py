"""
phase5_visualise.py  ─  Phase 5: Visualisation & Export
─────────────────────────────────────────────────────────
Sub-commands:
  view      - Open3D interactive viewer (default)
  pyvista   - PyVista polished viewer (optional)
  export    - Convert .ply → .glb (web) and/or .obj+.mtl (Unity)
  resume    - Load a saved scan session (.npz) and continue viewing
  save_session - Save point cloud + camera poses to a .npz session file

Usage:
  python phase5_visualise.py [view|pyvista|export|save_session|resume] [OPTIONS]

Examples:
  python phase5_visualise.py view --file scene.ply
  python phase5_visualise.py export --file scene.ply --glb --obj
  python phase5_visualise.py save_session --file scene.ply --poses poses.txt
  python phase5_visualise.py resume --session session.npz
"""

import argparse, os, sys, json

# ── Silence Qt noise before open3d (which also loads Qt) ─────────────────────
from utils import silence_qt
silence_qt()
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("mode",       nargs="?", default="view",
                    choices=["view", "pyvista", "export", "save_session", "resume"])
parser.add_argument("--file",     default="scene.ply",  help="Input .ply file")
parser.add_argument("--session",  default="session.npz",help="Session file for resume")
parser.add_argument("--poses",    default=None,          help="Camera pose file (optional .txt, 4x4 per line)")
parser.add_argument("--glb",      action="store_true",  help="Export as .glb (web/Three.js)")
parser.add_argument("--obj",      action="store_true",  help="Export as .obj+.mtl (Unity)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  Open3D import guard
# ─────────────────────────────────────────────────────────────────────────────
try:
    import open3d as o3d
except ImportError:
    print("open3d not installed.  Run:  pip install open3d")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Mode A - Open3D viewer with keyboard shortcuts
# ─────────────────────────────────────────────────────────────────────────────

def mode_view():
    ply = Path(args.file)
    if not ply.exists():
        print(f"File not found: {ply}\nRun phase4_reconstruct.py first.")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(str(ply))
    print(f"Loaded {len(pcd.points):,} points from '{ply}'")
    print("Controls:  Left-drag=rotate  Middle-drag=pan  Scroll=zoom")
    print("           r=reset view  p=screenshot  q=quit")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Scanner - Scene Viewer", width=1024, height=640)

    vis.add_geometry(pcd)

    # Add a small axis indicator at origin
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(frame)

    render = vis.get_render_option()
    if render is not None:
        render.background_color = np.array([0.05, 0.05, 0.1])
        render.point_size = 1.5
    else:
        print("[warn] Could not set render options (OpenGL init failed).")


    # Keyboard: 'r' → reset view
    def reset_view(vis):
        vis.reset_view_point(True)
        return False
    vis.register_key_callback(ord('R'), reset_view)

    # Keyboard: 'p' → save screenshot
    def screenshot(vis):
        path = "screenshot.png"
        vis.capture_screen_image(path)
        print(f"  Screenshot saved → {path}")
        return False
    vis.register_key_callback(ord('P'), screenshot)

    vis.run()
    vis.destroy_window()

# ─────────────────────────────────────────────────────────────────────────────
#  Mode B - PyVista viewer
# ─────────────────────────────────────────────────────────────────────────────

def mode_pyvista():
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista not installed.  Run:  pip install pyvista")
        sys.exit(1)

    ply = Path(args.file)
    cloud = pv.read(str(ply))
    print(f"PyVista loaded {cloud.n_points:,} points from '{ply}'")

    pl = pv.Plotter(window_size=[1024, 640])
    pl.background_color = "#0d0d1a"
    pl.add_mesh(cloud, point_size=2, render_points_as_spheres=False,
                scalars=cloud.points[:, 2],    # colour by Z height
                cmap="coolwarm", show_scalar_bar=True)
    pl.add_axes()
    pl.show(title="3D Scanner - PyVista Viewer")

# ─────────────────────────────────────────────────────────────────────────────
#  Mode C - Export (.glb / .obj)
# ─────────────────────────────────────────────────────────────────────────────

def mode_export():
    ply = Path(args.file)
    if not ply.exists():
        print(f"File not found: {ply}")
        sys.exit(1)

    # Detect if it's a mesh or point cloud
    mesh = o3d.io.read_triangle_mesh(str(ply))
    is_mesh = len(mesh.triangles) > 0

    if not is_mesh:
        print("  Input is a point cloud.  Converting to mesh via Ball-Pivoting …")
        pcd = o3d.io.read_point_cloud(str(ply))
        pcd.estimate_normals()
        radii = [0.02, 0.04, 0.08]
        mesh  = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        mesh.compute_vertex_normals()

    if args.glb:
        # trimesh is more reliable for glb export
        try:
            import trimesh as tm
            # Convert open3d mesh → trimesh
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            tm_mesh = tm.Trimesh(vertices=verts, faces=faces, process=False)
            glb_path = ply.with_suffix(".glb")
            tm_mesh.export(str(glb_path))
            print(f"  ✓  GLB saved → {glb_path}")
            _print_threejs_snippet(glb_path)
        except ImportError:
            print("  trimesh not installed.  Run:  pip install trimesh")

    if args.obj:
        obj_path = ply.with_suffix(".obj")
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        print(f"  ✓  OBJ saved → {obj_path}")
        _print_unity_instructions(obj_path)

    if not args.glb and not args.obj:
        print("  Specify --glb and/or --obj.  Nothing exported.")


def _print_threejs_snippet(glb_path: Path):
    print(f"""
──────────────────── Three.js HTML Snippet ────────────────────
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>3D Scene Viewer</title>
  <style>body {{ margin:0; overflow:hidden; background:#0d0d1a; }}</style>
</head>
<body>
<script type="importmap">
  {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.165/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165/examples/jsm/"}}}}
</script>
<script type="module">
  import * as THREE from 'three';
  import {{ GLTFLoader }}   from 'three/addons/loaders/GLTFLoader.js';
  import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(innerWidth, innerHeight);
  document.body.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.01, 100);
  camera.position.set(0, 1, 3);

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  scene.add(Object.assign(new THREE.DirectionalLight(0xffffff, 1), {{ position: new THREE.Vector3(5, 10, 7) }}));

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  new GLTFLoader().load('{glb_path.name}', gltf => scene.add(gltf.scene));
  window.addEventListener('resize', () => {{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  }});
  (function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }})();
</script>
</body>
</html>
───────────────────────────────────────────────────────────────
Copy the above HTML and place it alongside '{glb_path.name}'.
""")


def _print_unity_instructions(obj_path: Path):
    print(f"""
──────────────────── Unity Import Instructions ─────────────────
1. Copy '{obj_path}' (and any .mtl) into your project's Assets/Models/ folder.
2. Unity auto-imports it. Select the asset, set:
     Scale Factor      → 1 (if the model appears too small/large, adjust)
     Generate Lightmap UVs → ✓
3. Drag the prefab into the scene Hierarchy.
4. For AR: attach an 'AR Session' and 'AR Camera' from the XR Toolkit package.
   Set up an AR Plane Manager to anchor the scan on a detected surface.
   See: https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@5.1
────────────────────────────────────────────────────────────────
""")

# ─────────────────────────────────────────────────────────────────────────────
#  Mode D - Save session
# ─────────────────────────────────────────────────────────────────────────────

def mode_save_session():
    ply = Path(args.file)
    if not ply.exists():
        print(f"File not found: {ply}")
        sys.exit(1)

    pcd    = o3d.io.read_point_cloud(str(ply))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros((0, 3))

    poses = None
    if args.poses and Path(args.poses).exists():
        poses = np.loadtxt(args.poses)

    session_path = Path("session.npz")
    save_dict = {
        "points": points,
        "colors": colors,
        "source_ply": str(ply),
    }
    if poses is not None:
        save_dict["poses"] = poses

    np.savez_compressed(str(session_path), **save_dict)
    print(f"  ✓  Session saved → {session_path}")
    print(f"     {points.shape[0]:,} points, {len(save_dict.get('poses', []))} poses")

# ─────────────────────────────────────────────────────────────────────────────
#  Mode E - Resume session
# ─────────────────────────────────────────────────────────────────────────────

def mode_resume():
    session = Path(args.session)
    if not session.exists():
        print(f"Session file not found: {session}")
        sys.exit(1)

    data   = np.load(str(session), allow_pickle=True)
    points = data["points"]
    colors = data["colors"] if "colors" in data else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Resumed session from '{session}'  -  {len(points):,} points")
    if "source_ply" in data:
        print(f"  Original source: {data['source_ply']}")
    if "poses" in data:
        print(f"  Camera poses: {data['poses'].shape[0]}")

    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
        window_name=f"Resumed Session - {session.name}",
        width=1024, height=640
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    {
        "view":         mode_view,
        "pyvista":      mode_pyvista,
        "export":       mode_export,
        "save_session": mode_save_session,
        "resume":       mode_resume,
    }[args.mode]()
