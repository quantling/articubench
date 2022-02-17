from xml.dom import minidom
import numpy as np
from util import export_svgs
import os
import shutil
# calculate tongue height
# tongue is at line 11
# teath are mostly on line 17

# the reference teeth are calculated assuming the ultrasound tranceducer is
# attached vertically for an /a/ sound
REFERENCE_TEETH = np.array([
    [ -0.8998, -50.1746, 0.0],
    [ 13.8935, -45.563 , 0.0],
    [ 34.8429, -47.083 , 0.0],
    [ 43.2337, -53.3766, 0.0],
    [ 55.6199, -48.5904, 0.0],
    [ 76.0926, -50.0758, 0.0],
    [ 83.4936, -56.2977, 0.0],
    [ 95.2568, -51.4663, 0.0],
    [113.2068, -52.7686, 0.0],
    [119.8622, -58.9364, 0.0],
    [129.4752, -53.949 , 0.0],
    [143.7357, -54.9836, 0.0],
    [148.11  , -60.9859, 0.0],
    [156.9308, -55.941 , 0.0],
    [168.7142, -56.7959, 0.0],
    [170.1331, -62.5838, 0.0],
    [178.1814, -55.5879, 0.0],
    [185.8894, -56.1471, 0.0]]).T


def extract_highest_tongue_pos(svg_file):
    doc = minidom.parse(svg_file)  # parseString also exists
    polyline_point_strings = [path.getAttribute('points') for path
                    in doc.getElementsByTagName('polyline')]
    doc.unlink()

    tongue = polyline_point_strings[6]
    tongue = np.array([float(ff) for ff in tongue.split()])
    tongue.shape = (37, 2)
    # flip the vertical axis so that larger means higher
    tongue[:, 1] *= -1
    tongue = np.stack((tongue[:, 0], tongue[:, 1], np.zeros(37)), axis=1)
    tongue = tongue.T
    
    teeth = polyline_point_strings[12]
    teeth = np.array([float(ff) for ff in teeth.split()])
    teeth.shape = (18, 2)
    # flip the vertical axis so that larger means higher
    teeth[:, 1] *= -1
    teeth = np.stack((teeth[:, 0], teeth[:, 1], np.zeros(18)), axis=1)
    teeth = teeth.T


    # find rigid rotation and translation to align teeth to reference teeth
    R, t = rigid_transform_3D(teeth, REFERENCE_TEETH)

    # rotate and translate tongue accordingly and extract highest point in y-axis (vertical axis)
    highest_vert_point = (R @ tongue + t)[1, :].max()

    return highest_vert_point


def rigid_transform_3D(A, B):
    """
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    Source: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py

    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def tongue_heights_from_cps(cps):
    export_svgs(cps, path='svgs/', hop_length=5)
    tongue_pos = []

    for svg in np.sort(os.listdir("svgs")):
        tongue_pos += [extract_highest_tongue_pos(svg)]

    shutil.rmtree("svgs")

    return np.asarray(tongue_pos)
