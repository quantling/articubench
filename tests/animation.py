import ctypes
import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from articubench.util import cps_to_ema_and_mesh, normalize_cp, align_ema, scale_emas_to_vtl
from articubench.control_models import synth_paule_fast
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
data = pd.read_pickle('/home/bob/AndreWorkland/articubench/articubench/data/small.pkl')
#cps = data['reference_cp'].iloc[-1]
label = data['label'].iloc[-1]
print(label)

cps = synth_paule_fast(seq_length = data['len_cp'].iloc[-1],
                       target_semantic_vector=data['target_semantic_vector'].iloc[-1],
                        target_audio = data['target_sig'].iloc[-1],
                        sampling_rate= data['target_sr'].iloc[-1])
#plot_cps = normalize_cp(cps)
#plt.plot(plot_cps)
#plt.show()

speaker_file_name = ctypes.c_char_p('../resources/JD3.speaker'.encode())
segment_file_name = ctypes.c_char_p(f'{label}.seg'.encode())
gesture_file_name = ctypes.c_char_p(f'{label}.ges'.encode())
file_name = ctypes.c_char_p(f'{label}'.encode())
path_name = ctypes.c_char_p(f'Meshes/{label}'.encode())


#we get our ema and meshes

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



def visualize_vtl_animation(mesh_dir, ema_file, frame_time=0.2):
    """
    Visualize VTL meshes and EMA points animation
    
    Args:
        mesh_dir: Directory containing the mesh OBJ files
        ema_file: Path to the EMA points file
        frame_time: Time between frames in seconds
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VTL Animation Viewer")
    
    # Get sorted list of mesh files and load EMA data once
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))
    ref_ema_ttip = data['reference_ema_TT'].iloc[-1]
    ref_ema_tmiddle = data['reference_ema_TB'].iloc[-1]

    ema_points = np.loadtxt(ema_file, skiprows=1)

    _, ref_ema_ttip = align_ema(ema_points, ref_ema_ttip)
    _, ref_ema_tmiddle = align_ema(ema_points, ref_ema_tmiddle)
    #ref_ema_tt = scale_emas_to_vtl(ref_ema_tt, ema_point)
    #ref_ema_tb = scale_emas_to_vtl(ref_ema_tb)
    if not mesh_files:
        print("No mesh files found in directory!")
        return
    
    if len(mesh_files) != len(ref_ema_ttip):
        print(f"Warning: Number of meshes ({len(mesh_files)}) doesn't match EMA frames ({len(ref_ema_tt)})")
    
    meshes = []
    for mesh_file in tqdm(sorted(mesh_files, key=lambda x: int(os.path.basename(x).split(".")[0].removeprefix(f"{label}")))):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    
    current_mesh = meshes[0]
    frame_ema = ema_points[0]
    tback = o3d.geometry.PointCloud()
    tmiddle= o3d.geometry.PointCloud()
    ttip = o3d.geometry.PointCloud()
    ref_tmiddle= o3d.geometry.PointCloud()
    ref_ttip = o3d.geometry.PointCloud()
    
    tback.points = o3d.utility.Vector3dVector([ema_points[0, 1:4]])  
    tmiddle.points = o3d.utility.Vector3dVector([ema_points[0, 4:7]])
    ttip.points = o3d.utility.Vector3dVector([ema_points[0, 7:10]])
    ref_tmiddle.points = o3d.utility.Vector3dVector([ref_ema_ttip[0]])  
    ref_ttip.points = o3d.utility.Vector3dVector([ref_ema_tmiddle[0]]) 
    
    tback.paint_uniform_color([1, 0, 0]) #back is red
    tmiddle.paint_uniform_color([0, 1, 0]) #middle is green
    ttip.paint_uniform_color([0, 0, 1]) #tip is blue
    ref_tmiddle.paint_uniform_color([0.2, 1, 0]) #middle is green
    ref_ttip.paint_uniform_color([0.2, 0, 1]) #tip is blue
    
    #connection lines
    lines_middle = o3d.geometry.LineSet()
    lines_tip = o3d.geometry.LineSet()

    lines_middle.points = o3d.utility.Vector3dVector(np.vstack([
        ema_points[0, 4:7],  # tmiddle
        ref_ema_ttip[0]        # ref_tmiddle
    ]))
    lines_tip.points = o3d.utility.Vector3dVector(np.vstack([
        ema_points[0, 7:10], # ttip
        ref_ema_tmiddle[0]        # ref_ttip
    ]))
    
    # Define line connections (connect point 0 to point 1)
    lines_middle.lines = o3d.utility.Vector2iVector([[0, 1]])
    lines_tip.lines = o3d.utility.Vector2iVector([[0, 1]])
    
    # Set line colors
    lines_middle.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
    lines_tip.colors = o3d.utility.Vector3dVector([[0, 0, 1]])     # Blue
    
    # Add text for distances
    #dist_text_ttip = o3d.visualization.TextGeometry("Distance: ")
    #dist_text_ttip.font_size = 16
    #dist_text_ttip.origin = np.array([-5, 5, 0]) 


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
    #vis.add_geometry(create_coordinate_frame())

    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -.8])
    ctr.set_up([0, .8, 0])
    
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
                
                # Update EMA points
                frame_ema = ema_points[frame_idx]
                tback.points = o3d.utility.Vector3dVector([frame_ema[1:4]])
                tmiddle.points = o3d.utility.Vector3dVector([frame_ema[4:7]])
                ttip.points = o3d.utility.Vector3dVector([frame_ema[7:10]])
                ref_tmiddle.points = o3d.utility.Vector3dVector([ref_ema_tmiddle[frame_idx]])
                ref_ttip.points = o3d.utility.Vector3dVector([ref_ema_ttip[frame_idx]])
                

                lines_middle.points = o3d.utility.Vector3dVector(np.vstack([
                    frame_ema[4:7],      # tmiddle
                    ref_ema_tmiddle[frame_idx] # ref_tmiddle
                ]))
                lines_tip.points = o3d.utility.Vector3dVector(np.vstack([
                    frame_ema[7:10],     # ttip  
                    ref_ema_ttip[frame_idx] # ref_ttip
                ]))
                
                # Calculate distances
                curr_dist = np.linalg.norm(frame_ema[7:10] - ref_ema_ttip[frame_idx])
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

                last_update = current_time
                frame_idx += 1
            
            vis.poll_events()
            vis.update_renderer()
            
    except KeyboardInterrupt:
        print("Animation stopped by user")
    finally:
        vis.destroy_window()
        shutil.rmtree(DIR + '/Meshes')  

if __name__ == "__main__":
    mesh_dir = DIR +f"/Meshes/{label}-meshes/"
    ema_file = DIR + f"/Meshes/{label}-ema.txt"
    
    visualize_vtl_animation(mesh_dir, ema_file)
