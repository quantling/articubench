import ctypes
import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from articubench.util import cps_to_ema_and_mesh, normalize_cp, align_ema, scale_emas_to_vtl
from articubench.control_models import synth_paule_fast
import cv2
import numpy as np
from PIL import Image
import io

DIR = os.path.dirname(__file__)
MESH_DIR = os.path.join(DIR, 'Meshes')




#then we add the open3d code to visualize the animation
import open3d as o3d
import numpy as np
import glob
import os
import time

def create_coordinate_frame(size=1.0):
    """Creates a coordinate frame for reference"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_trail_points(points_history, colors, alpha_values):
    """
    Create a point cloud for trailing points with fading colors
    
    Args:
        points_history: List of point coordinates
        colors: Base colors for points
        alpha_values: Alpha values for each point (fading effect)
    """
    if not points_history:
        return None
        
    trail = o3d.geometry.PointCloud()
    trail.points = o3d.utility.Vector3dVector(points_history)
    
    # Create fading colors
    faded_colors = []
    for i, alpha in enumerate(alpha_values):
        faded_color = colors.copy()
        faded_color.append(alpha)  # Add alpha value
        faded_colors.append(faded_color)
    
    trail.colors = o3d.utility.Vector3dVector(faded_colors)
    return trail

def visualize_vtl_animation(mesh_dir, ema_file, scale=False, frame_time=0.5, trail_length=5, export=True,width=1024, height=768):
    """
    Visualize VTL meshes and EMA points animation
    
    Args:
        mesh_dir: Directory containing the mesh OBJ files
        ema_file: Path to the EMA points file
        frame_time: Time between frames in seconds
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VTL Animation Viewer", width=width, height=height)
    
    # Get sorted list of mesh files and load EMA data once
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    ref_ema_ttip = data['reference_ema_TT']
    ref_ema_tmiddle = data['reference_ema_TB']

    ema_points = np.loadtxt(ema_file, skiprows=1)

    #ref emas are notoriously like 2 off ._.
    _, ref_ema_ttip = align_ema(ema_points, ref_ema_ttip)
    _, ref_ema_tmiddle = align_ema(ema_points, ref_ema_tmiddle)

    if scale: #x_offset=8, y_offset=0.3 z_offset=-0.3
        ref_ema_ttip = scale_emas_to_vtl(ref_ema_ttip, x_offset=9)
        ref_ema_tmiddle = scale_emas_to_vtl(ref_ema_tmiddle, z_offset=-0.6)
        
    if not mesh_files:
        print("No mesh files found in directory!")
        return
    
    if len(mesh_files) != len(ref_ema_ttip):
        print(f"Warning: Number of meshes ({len(mesh_files)}) doesn't match EMA frames ({len(ref_ema_ttip)})")
    
    meshes = []
    for mesh_file in tqdm(sorted(mesh_files, key=lambda x: int(os.path.basename(x).split(".")[0].removeprefix(f"{label}")))):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    
    current_mesh = meshes[0]

    tback = o3d.geometry.PointCloud()
    tmiddle= o3d.geometry.PointCloud()
    ttip = o3d.geometry.PointCloud()
    ref_tmiddle= o3d.geometry.PointCloud()
    ref_ttip = o3d.geometry.PointCloud()
    
    tback.points = o3d.utility.Vector3dVector([ema_points[0, 1:4]])  
    tmiddle.points = o3d.utility.Vector3dVector([ema_points[0, 4:7]])
    ttip.points = o3d.utility.Vector3dVector([ema_points[0, 7:10]])
    ref_tmiddle.points = o3d.utility.Vector3dVector([ref_ema_tmiddle[0]])  
    ref_ttip.points = o3d.utility.Vector3dVector([ref_ema_ttip[0]]) 
    
    tback.paint_uniform_color([1, 0, 0]) #back is red
    tmiddle.paint_uniform_color([0, 1, 0]) #middle is green
    ttip.paint_uniform_color([0, 0, 1]) #tip is blue
    ref_tmiddle.paint_uniform_color([0.5, 1, 0.5]) #middle is greenish
    ref_ttip.paint_uniform_color([0.5, 0.5, 1]) #tip is blueish
    
    #init lines
    lines_middle = o3d.geometry.LineSet()
    lines_tip = o3d.geometry.LineSet()

    #set initial line points where our ema and ref_emas are
    lines_middle.points = o3d.utility.Vector3dVector(np.vstack([
        ema_points[0, 4:7],  
        ref_ema_tmiddle[0]       
    ]))
    lines_tip.points = o3d.utility.Vector3dVector(np.vstack([
        ema_points[0, 7:10],
        ref_ema_ttip[0]      
    ]))
    
    #initial line connections
    lines_middle.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines_tip.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines_middle.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
    lines_tip.colors = o3d.utility.Vector3dVector([[0, 0, 1]])     # Blue
    
    # Initialize history for trailing points
    ttip_history = []
    tmiddle_history = []
    ref_ttip_history = []
    ref_tmiddle_history = []

    vis.add_geometry(current_mesh)
    vis.add_geometry(tback)
    vis.add_geometry(tmiddle)
    vis.add_geometry(ttip)
    vis.add_geometry(ref_tmiddle)
    vis.add_geometry(ref_ttip)
    vis.add_geometry(lines_middle)
    vis.add_geometry(lines_tip)
    #vis.add_geometry(dist_text_ttip)
    # Add coordinate frame, useful for reference x(red), y(green), z(blue)
    vis.add_geometry(create_coordinate_frame())
    
    # Create trailing point clouds (they'll be updated in the loop)
    trail_ttip = o3d.geometry.PointCloud()
    trail_tmiddle = o3d.geometry.PointCloud()
    trail_ref_ttip = o3d.geometry.PointCloud()
    trail_ref_tmiddle = o3d.geometry.PointCloud()
    vis.add_geometry(trail_ttip)
    vis.add_geometry(trail_tmiddle)
    vis.add_geometry(trail_ref_ttip)
    vis.add_geometry(trail_ref_tmiddle)
    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.9)
    ctr.set_front([0.2, 0.8, -.8])
    ctr.set_up([0.1, 1, 0])
    if export:
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename='output.mp4', fourcc=fourcc, fps=30, frameSize=(1024, 768))

    try:
        last_update = time.time()
        frame_idx = 0
        
        while frame_idx < len(mesh_files):
            current_time = time.time()
            
            if current_time - last_update >= frame_time:
                # Update mesh
                new_mesh = meshes[frame_idx]
                current_mesh.vertices = new_mesh.vertices
                current_mesh.triangles = new_mesh.triangles
                current_mesh.compute_vertex_normals()
                
                # Get current EMA points
                current_ttip = ema_points[frame_idx, 7:10]
                current_tmiddle = ema_points[frame_idx, 4:7]
                current_ref_ttip = ref_ema_ttip[frame_idx]
                current_ref_tmiddle = ref_ema_tmiddle[frame_idx]
                
                # Update EMA points
                tback.points = o3d.utility.Vector3dVector([ema_points[frame_idx, 1:4]])
                tmiddle.points = o3d.utility.Vector3dVector([current_tmiddle])
                ttip.points = o3d.utility.Vector3dVector([current_ttip])
                ref_tmiddle.points = o3d.utility.Vector3dVector([current_ref_tmiddle])
                ref_ttip.points = o3d.utility.Vector3dVector([current_ref_ttip])
                
                # Update connection lines
                lines_middle.points = o3d.utility.Vector3dVector(np.vstack([
                    current_tmiddle,
                    current_ref_tmiddle
                ]))
                lines_tip.points = o3d.utility.Vector3dVector(np.vstack([
                    current_ttip,
                    current_ref_ttip
                ]))
                
                # Update history for trailing points
                ttip_history.append(current_ttip)
                tmiddle_history.append(current_tmiddle)
                ref_ttip_history.append(current_ref_ttip)
                ref_tmiddle_history.append(current_ref_tmiddle)
                
                # trim history!
                if len(ttip_history) > trail_length:
                    ttip_history = ttip_history[-trail_length:]
                    tmiddle_history = tmiddle_history[-trail_length:]
                    ref_ttip_history = ref_ttip_history[-trail_length:]
                    ref_tmiddle_history = ref_tmiddle_history[-trail_length:]
                
                # Create alpha values for fading effect (oldest = most transparent)
                alpha_values = np.linspace(0.1, 1.0, len(ttip_history))
                
                # Update trail point clouds
                if ttip_history:
                    # Base colors for each trail
                    ttip_color = [0, 0, 1]  # Blue
                    tmiddle_color = [0, 1, 0]  # Green
                    ref_ttip_color = [0.5, 0.5, 1]  # Light blue
                    ref_tmiddle_color = [0.5, 1, 0.5]  # Light green
                    
                    # Create fading colors - Open3D expects RGB without alpha in the colors attribute
                    ttip_colors = np.array([[*ttip_color] for _ in alpha_values])
                    tmiddle_colors = np.array([[*tmiddle_color] for _ in alpha_values])
                    ref_ttip_colors = np.array([[*ref_ttip_color] for _ in alpha_values])
                    ref_tmiddle_colors = np.array([[*ref_tmiddle_color] for _ in alpha_values])
                    
                    # update trail points after converting to np arrays
                    ttip_history_array = np.array(ttip_history)
                    tmiddle_history_array = np.array(tmiddle_history)
                    ref_ttip_history_array = np.array(ref_ttip_history)
                    ref_tmiddle_history_array = np.array(ref_tmiddle_history)
                    
                    # Only update if we have valid points or we get error
                    if len(ttip_history_array) > 0:
                        trail_ttip.colors = o3d.utility.Vector3dVector(ttip_colors)
                        trail_tmiddle.colors = o3d.utility.Vector3dVector(tmiddle_colors)
                        trail_ref_ttip.colors = o3d.utility.Vector3dVector(ref_ttip_colors)
                        trail_ref_tmiddle.colors = o3d.utility.Vector3dVector(ref_tmiddle_colors)
                        trail_ttip.points = o3d.utility.Vector3dVector(ttip_history_array)
                        trail_tmiddle.points = o3d.utility.Vector3dVector(tmiddle_history_array)
                        trail_ref_ttip.points = o3d.utility.Vector3dVector(ref_ttip_history_array)
                        trail_ref_tmiddle.points = o3d.utility.Vector3dVector(ref_tmiddle_history_array)
         

                # Calculate distances
                curr_dist = np.linalg.norm(ema_points[frame_idx, 7:10] - ref_ema_ttip[frame_idx])
                print(curr_dist)
                #total_dist += curr_dist
                #avg_dist = total_dist / (frame_idx + 1)
                

                # Update distance text
                #dist_text_ttip.text = f"Tongue Tip, Current dist: {curr_dist:.2f}\nAvg dist: {avg_dist:.2f}"
                
                # Update visualization
                vis.update_geometry(current_mesh)
                vis.update_geometry(tback)
                vis.update_geometry(tmiddle)
                vis.update_geometry(ttip)
                vis.update_geometry(ref_tmiddle)
                vis.update_geometry(ref_ttip)
                vis.update_geometry(lines_middle)
                vis.update_geometry(lines_tip)
                #vis.update_geometry(dist_text_ttip)
                vis.update_geometry(trail_ttip)
                vis.update_geometry(trail_tmiddle)
                vis.update_geometry(trail_ref_ttip)
                vis.update_geometry(trail_ref_tmiddle)

                if export:
                    # Capture frame
                    img = vis.capture_screen_float_buffer()
                    img = np.asarray(img)
                    img = (img * 255).astype(np.uint8)
                    
                    # Convert to BGR for OpenCV
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    out.write(img)

                last_update = current_time
                frame_idx += 1
            
            vis.poll_events()
            vis.update_renderer()
            
    except KeyboardInterrupt:
        print("Animation stopped by user")
    finally:
        if export:
            out.release()
        vis.destroy_window()
        shutil.rmtree(DIR + '/Meshes')  

if __name__ == "__main__":
    if os.path.exists(MESH_DIR):
        print("Removing old meshes...")
        shutil.rmtree(MESH_DIR)

    data = pd.read_pickle('./articubench/data/small.pkl').iloc[-1]
    label = data['label']
    print(label)
    print(data)
    cps = synth_paule_fast(seq_length = data['len_cp'],
                        target_semantic_vector=data['target_semantic_vector'],
                            target_audio = data['target_sig'],
                            sampling_rate= data['target_sr'])

    cps_to_ema_and_mesh(cps, f'{label}', path=DIR + '/Meshes')
    mesh_dir = os.path.join(MESH_DIR, f"{label}-meshes")
    ema_file = os.path.join(MESH_DIR, f"{label}-ema.txt")

    visualize_vtl_animation(mesh_dir, ema_file, scale=True, export=True)
