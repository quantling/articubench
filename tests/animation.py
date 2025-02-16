import ctypes
import os
import sys
import shutil
import pandas as pd
#First we create ema and meshes like in backendDev

DIR = os.path.dirname(__file__)
# load vocaltractlab binary
PREFIX = "lib"
SUFFIX = ""
if sys.platform.startswith("linux"):
    SUFFIX = ".so"
elif sys.platform.startswith("win32"):
    PREFIX = ""
    SUFFIX = ".dll"
elif sys.platform.startswith("darwin"):
    SUFFIX = ".dylib"
VTL = ctypes.cdll.LoadLibrary(
    os.path.join(DIR, f"vocaltractlab_api/{PREFIX}VocalTractLabApi{SUFFIX}")
)
# initialize vtl
speaker_file_name = ctypes.c_char_p(
    os.path.join(DIR, "vocaltractlab_api/JD3.speaker").encode()
)
#initialize file and path names
data = pd.read_pickle('/home/andre/articubench/articubench/data/tiny.pkl')
cps = data['reference_cp'].iloc[0]
label = data['label'].iloc[0]

speaker_file_name = ctypes.c_char_p('../resources/JD3.speaker'.encode())
segment_file_name = ctypes.c_char_p(f'{label}.seg'.encode())
gesture_file_name = ctypes.c_char_p(f'{label}.ges'.encode())
file_name = ctypes.c_char_p(f'{label}'.encode())
path_name = ctypes.c_char_p(f'Meshes/{label}'.encode())


#we get our ema and meshes
from articubench.util import cps_to_ema_and_mesh
cps_to_ema_and_mesh(cps, f'{label}', path=DIR + '/Meshes')


#then we add the open3d code to visualize the animation
import open3d as o3d
import numpy as np
import glob
import os
import time

def create_coordinate_frame(size=1.0):
    """Creates a coordinate frame for reference"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def load_mesh_and_emas(mesh_path, ema_file, frame_index):
    """Load mesh and EMA points"""
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # Create EMA points as spheres
    ema_points = np.loadtxt(ema_file, skiprows=1)
    pcd_tback = o3d.geometry.PointCloud()
    pcd_tmiddle= o3d.geometry.PointCloud()
    pcd_ttip = o3d.geometry.PointCloud()
    pcd_tback.points = o3d.utility.Vector3dVector([ema_points[frame_index, 1:4]])  # Assuming first column is time
    pcd_tmiddle.points = o3d.utility.Vector3dVector([ema_points[frame_index, 4:7]])  # Assuming first column is time
    pcd_ttip.points = o3d.utility.Vector3dVector([ema_points[frame_index, 7:10]])  # Assuming first column is time
    # Color EMA points red
    pcd_tback.paint_uniform_color([1, 0, 0]) #back is red
    pcd_tmiddle.paint_uniform_color([0, 1, 0]) #middle is green
    pcd_ttip.paint_uniform_color([0, 0, 1]) #tip is blue
    
    return mesh, (pcd_tback, pcd_tmiddle, pcd_ttip)

def visualize_vtl_animation(mesh_dir, ema_file, frame_time=0.1):
    """
    Visualize VTL meshes and EMA points animation
    
    Args:
        mesh_dir: Directory containing the mesh OBJ files
        ema_file: Path to the EMA points file
        frame_time: Time between frames in seconds
    """
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VTL Animation Viewer")
    
    # Get sorted list of mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    
    if not mesh_files:
        print("No mesh files found in directory!")
        return
        
    # Load first mesh to setup scene
    current_mesh, (tback, tmiddle, ttip) = load_mesh_and_emas(mesh_files[0], ema_file, 0)
    
    # Add geometries to visualizer
    vis.add_geometry(current_mesh)
    vis.add_geometry(tback)
    vis.add_geometry(tmiddle)
    vis.add_geometry(ttip)
    
    # Add coordinate frame, useful for reference x(red), y(green), z(blue)
    #vis.add_geometry(create_coordinate_frame())
    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    
    try:
        for frame_idx, mesh_file in enumerate(mesh_files):
            # Update mesh
            new_mesh, (new_tback, new_tmiddle, new_ttip) = load_mesh_and_emas(mesh_file, ema_file, frame_idx)
            current_mesh.vertices = new_mesh.vertices
            current_mesh.triangles = new_mesh.triangles
            current_mesh.compute_vertex_normals()
            
            # Update EMA points
            tback.points = new_tback.points
            tmiddle.points = new_tmiddle.points
            ttip.points = new_ttip.points
            
            # Update visualization
            vis.update_geometry(current_mesh)
            vis.update_geometry(tback)
            vis.update_geometry(tmiddle)
            vis.update_geometry(ttip)
            vis.poll_events()
            vis.update_renderer()
            
            time.sleep(frame_time)
            
    except KeyboardInterrupt:
        print("Animation stopped by user")
        shutil.rmtree(DIR + '/Meshes') #cleanup mesh files
    finally:
        vis.destroy_window()

if __name__ == "__main__":
    # Example usage
    mesh_dir = DIR +f"/Meshes/{label}-meshes/"
    ema_file = DIR + f"/Meshes/{label}-ema.txt"
    
    visualize_vtl_animation(mesh_dir, ema_file)
