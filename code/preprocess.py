import laspy
from pyproj import CRS
import os

def preprocess(input_las):
    """
    Function that clips a large LAZ file into 500m*500m and remove classification and RGB values

    Input:
        input_file
    Output:
        preprocessed LAZ file
    Return:
        preprocessed LAZ data
    """
    # 1) find a center point
    mins = input_las.header.min
    maxs = input_las.header.max
    center = (mins + maxs) / 2

    # 2) set a mask box
    box_x_min = center[0] - 250
    box_x_max = center[0] + 250
    box_y_min = center[1] - 250
    box_y_max = center[1] + 250

    # 3) set a header of an output file
    output_header = laspy.LasHeader(point_format=input_las.header.point_format, version=input_las.header.version)
    output_header.scales = input_las.header.scales
    output_header.offset = input_las.header.offsets
    output_header.add_crs(CRS.from_epsg(28992))

    # 4) write the output file
    os.makedirs("data", exist_ok=True)  # ensure folder exists
    output_file_nm = 'data/out.laz'

    with laspy.open(output_file_nm, "w", header=output_header) as out_las:
        # use chunk_iterator to read the large file
        for points in input_las.chunk_iterator(1_000_000):
            # 5) remove classification and RGB values
            points.classification[:] = 0

            if hasattr(points, "red") and hasattr(points, "green") and hasattr(points, "blue"):
                points.red[:] = 0
                points.green[:] = 0
                points.blue[:] = 0

            # 6) filter points with mask
            mask_box = (points.x >= box_x_min) & (points.x <= box_x_max) & \
                       (points.y >= box_y_min) & (points.y <= box_y_max)
            filtered_points = points[mask_box]
            
            out_las.write_points(filtered_points)

            # delete it to reduce memory usage
            del filtered_points

        print("clipping done\n")

    return output_file_nm
