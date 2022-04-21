import os
import tempfile

from xml.dom import minidom
import numpy as np

from .util import export_svgs, calculate_roll_pitch_yaw, rigid_transform_3d
# calculate tongue height
# tongue is at line 11
# teath are mostly on line 17

# the reference teeth are calculated assuming the ultrasound tranceducer is
# attached vertically for an /o/ sound
REFERENCE_TEETH = np.array([
       [ -5.5254, 58.7969,   0.],
       [  9.4225, 54.7141,   0.],
       [ 30.3047, 56.9778,   0.],
       [ 38.4664, 63.5657,   0.],
       [ 51.0149, 59.2228,   0.],
       [ 71.4219, 61.435 ,   0.],
       [ 78.5971, 67.9161,   0.],
       [ 90.5246, 63.5058,   0.],
       [108.417 , 65.4454,   0.],
       [114.8489, 71.8459,   0.],
       [124.6331, 67.2033,   0.],
       [138.8478, 68.7442,   0.],
       [143.0059, 74.8982,   0.],
       [152.0005, 70.17  ,   0.],
       [163.7461, 71.4433,   0.],
       [164.9583, 77.2779,   0.],
       [173.2503, 70.5725,   0.],
       [180.9335, 71.4054,   0.]]).T


def highest_tongue_position(svg_file):
    """
    Returns the highest tongue_point in respect to fixed lower teeth in the /o/
    orientation together with the rotation matrix and the translation vector.

    """
    doc = minidom.parse(svg_file)  # parseString also exists
    polyline_point_strings = [path.getAttribute('points') for path
                    in doc.getElementsByTagName('polyline')]
    doc.unlink()

    tongue = polyline_point_strings[6]
    tongue = np.array([float(ff) for ff in tongue.split()])
    tongue.shape = (37, 2)

    tongue = np.stack((tongue[:, 0], tongue[:, 1], np.zeros(37)), axis=1)
    tongue = tongue.T
    
    teeth = polyline_point_strings[12]
    teeth = np.array([float(ff) for ff in teeth.split()])
    teeth.shape = (18, 2)

    teeth = np.stack((teeth[:, 0], teeth[:, 1], np.zeros(18)), axis=1)
    teeth = teeth.T

    # find best rotation and translation
    rotation, translation = rigid_transform_3d(teeth, REFERENCE_TEETH)

    # rotate and translate tongue accordingly
    rotated_tongue = (rotation @ tongue + translation)

    # Extract highest point in y-axis (vertical axis) which corresponds to the
    # minimal point, as the in visual coordinates the y-coordinate increases
    # from top to bottom.
    highest_vert_point_index = rotated_tongue[1, :].argmin()
    highest_vert_point = rotated_tongue[:, highest_vert_point_index]

    return highest_vert_point, rotation, translation


def tongue_height_from_cps(cps):
    with tempfile.TemporaryDirectory(prefix='python_articubench_') as path:
        # extract tongue height with roughly 80 Hz \approx 1 : 550 / 44100
        export_svgs(cps, path=path, hop_length=5)
        tongue_pos = []

        for svg in np.sort(os.listdir(path)):
            highest_point, _, _ = highest_tongue_position(os.path.join(path, svg))
            y_coord = highest_point[1]
            y_coord *= -1  # flip to make a larger value a higher point
            tongue_pos.append(y_coord)

    return np.asarray(tongue_pos)


def visualize_highest_points(cps, *, target_dir='highest_svgs'):
    import svgutils
    with tempfile.TemporaryDirectory(prefix='python_articubench_') as path:
        # extract tongue height with roughly 80 Hz \approx 1 : 550 / 44100
        export_svgs(cps, path=path, hop_length=5)
        tongue_pos = []

        for svg_name in np.sort(os.listdir(path)):
            source_svg = os.path.join(path, svg_name)
            target_svg = os.path.join(target_dir, svg_name)

            svg = svgutils.transform.fromfile(source_svg)
            vtl_plot = svg.getroot()

            point, R, t = test_highest_point_with_reference_teeth(source_svg)
            roll, pitch, yaw = calculate_roll_pitch_yaw(R)

            vtl_plot.rotate(yaw)
            vtl_plot.moveto(float(t[0]), float(t[1]))

            hradius = np.array([20.0, 0.0])
            vradius = np.array([0.0, 6.0])
            hpoints = (list(point[:2] - hradius), list(point[:2] + hradius))
            vpoints = (list(point[:2] - vradius), list(point[:2] + vradius))
            hline = svgutils.transform.LineElement(hpoints, width=2.0, color='red')
            vline = svgutils.transform.LineElement(vpoints, width=2.0, color='red')

            figure = svgutils.transform.SVGFigure(svg.width, svg.height)
            figure.append([vtl_plot, hline, vline])
            figure.save('temp.svg')

            # fix viewBox
            with open('temp.svg', 'rt') as in_file:
                with open(target_svg, 'wt') as out_file:
                    for ii, line in enumerate(in_file):
                        if ii == 1:  # second line
                            out_file.write('<svg width="768" height="576" viewBox="-170 -90 520 390" version="1.1" xmlns="http://www.w3.org/2000/svg">\n')
                            continue
                        out_file.write(line)

    print(r"""
          To create a video from the svg files run:

            /usr/bin/ffmpeg -r 80.181818 -width 720 -i {target_dir}/tract%05d.svg -i SOURCE_AUDIO.flac -filter:a "volume=7.0" VIDEO_80HZ.webm
            /usr/bin/ffmpeg -i VIDEO_80HZ.webm -r 60 VIDEO_60Hz.webm
          """)

