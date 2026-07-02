import time
import laspy
import numpy as np
import scipy.spatial
import scipy.stats
import startinpy
import rasterio
import os
from pyproj import CRS

# global variables
MOVABLE = 1
UNMOVABLE = 0

class ClothPoints:
    def __init__(self, xs, ys, zs, label):
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.label = label        # MOVABLE or UNMOVABLE
        self.cp_idx = None        # index of the nearest point
        self.cp_dist = None       # distance between the corresponding point
        self.cp_z = None          # height of the corresponding point
        self.indices_list = None  # points that are located within cp_dist

    def pts_copy(self):
        return ClothPoints(self.xs.copy(), self.ys.copy(), self.zs.copy(), self.label.copy())

    def get_cp_dist_idx(self, tree):
        """
        Function that updates the index and distance of the CP

        Input:
            tree: KDtree of the inversed original points
        """
        xy_coords = np.column_stack((self.xs.ravel(), self.ys.ravel()))
        self.cp_dist, self.cp_idx = tree.query(xy_coords, k=1)  # k=1 : select only one nearest point
        eps = 1e-6
        self.indices_list = tree.query_ball_point(xy_coords, self.cp_dist+eps)

    def get_cp_z(self, las_pts_z):
        """
        Function that updates the z value of the CP

        Input:
            las_pts_z: z values of the inversed original points
        """
        cp_z_values = []
        
        for indices in self.indices_list:
            if len(indices) > 0:
                local_zs = []
                for idx in indices:
                    local_zs.append(las_pts_z[idx])
                local_max_z = np.max(np.array(local_zs))
                cp_z_values.append(local_max_z)
            else:
                cp_z_values.append(-9999.0)

        self.cp_z = np.array(cp_z_values)

    def reshape(self):
        """
        Function to reshape the array to align with the cloth grid
        """
        self.cp_idx = self.cp_idx.reshape(self.xs.shape)
        self.cp_dist = self.cp_dist.reshape(self.xs.shape)
        self.cp_z = self.cp_z.reshape(self.xs.shape)

    def external_force(self, i, j, z_prev, dT, G=1.0):
        """
        Function to apply the displacement by the external force

        Input:
            i: row index of cloth particle
            j: column index of cloth particle
            dT: time step, which controls the displacement of particles from gravity
            G: gravity
               -> default = 1.0
        """
        z_cur = self.zs[i][j]

        z_new = 2 * z_cur - z_prev - G * (dT ** 2)
        self.zs[i][j] = z_new

    def intersection_check(self, i, j):
        """
        Function to update the label by comparing new z value with the z value of the CP

        Input:
            i: row index of cloth particle
            j: column index of cloth particle
        """
        # if the cloth particle is equal or below than the CP, update the z value and change the label
        if self.zs[i][j] <= self.cp_z[i][j]:
            self.zs[i][j] = self.cp_z[i][j]  # update the z value
            self.label[i][j] = 0             # change the label to UNMOVABLE

    def internal_force(self, RI=2):
        """
        Function to apply the displacement by the internal force

        Input:
            i: row index of cloth particle
            j: column index of cloth particle
            RI: rigidness of the cloth
                -> default = 2
        """
        # grid size
        row_n = len(self.xs)
        col_n = len(self.xs[0])

        while RI > 0:
            # execute only if the particle is movable
            for i in range(0, row_n):
                for j in range(0, col_n):
                    if i < row_n - 1:
                        self.internal_force_each_neighbour(i, j, i + 1, j)  # bottom
                    if j < col_n - 1:
                        self.internal_force_each_neighbour(i, j, i, j + 1)  # right

            RI -= 1

    def internal_force_each_neighbour(self, i, j, nbr_row_i, nbr_col_j):
        """
        Function to calculate the displacement by the internal force

        Input:
            i: row index of cloth particle
            j: column index of cloth particle
            nbr_position: position of the neighbour(t/r/b/l)

        Return:
            displacement vector
        """
        nbr_label = self.label[nbr_row_i][nbr_col_j]
        nbr_z = self.zs[nbr_row_i][nbr_col_j]

        p_label = self.label[i][j]
        p_z = self.zs[i][j]

        if not np.isclose(p_z, nbr_z):
            disp_v = 1 / 2 * p_label * (nbr_z - p_z)
            # if both are movable, move both in opposite direction
            if p_label == 1 and nbr_label == 1:
                self.zs[i][j] += disp_v
                self.zs[nbr_row_i][nbr_col_j] -= disp_v
            # if the neighbour is unmovable, only move the particle
            elif p_label == 1 and nbr_label == 0:
                self.zs[i][j] += disp_v
            # if the particle is unmovable, only move the neighbour
            elif p_label == 0 and nbr_label == 1:
                self.zs[nbr_row_i][nbr_col_j] -= disp_v


def init_cloth(x_min, x_max, y_min, y_max, z_max, GR):
    """
    Function that initializes the cloth

    Input:
        x_min: minimum x value of the original las data
        x_max: maximum x value of the original las data
        y_min: minimum y value of the original las data
        y_max: maximum y value of the original las data
        z_max: maximum z value of the original las data
        GR: grid resolution

    Return:
        grid_x: x values of the cloth
        grid_y: y values of the cloth
        grid_z: z values of the cloth
        grid_label: labels of each cloth particle (MOVABLE or UNMOVABLE)
    """
    cloth_z = z_max + 1.0

    xs = np.arange(x_min, x_max + GR, GR)
    ys = np.arange(y_min, y_max + GR, GR)

    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = np.full_like(grid_x, cloth_z)
    grid_label = np.full_like(grid_x, MOVABLE)

    return grid_x, grid_y, grid_z, grid_label


def cloth_displacement(input_file, itr_num, GR, dT, max_delta):
    """
    Function to displace the cloth particle

    Input:
        input_file: original las file
        itr_num: iteration number
        GR: grid resolution
        dT: time step for the external force
        max_delta: CSF tolerance espilon_zmax to stop the iterations

    Return:
        cloth_pts: displaced cloth particles
    """
    # 1) invert the original las dataset
    inv_las = laspy.read(input_file)
    inv_las.z *= -1

    # 2) create a cloth
    x_min = float(inv_las.header.min[0])
    x_max = float(inv_las.header.max[0])
    y_min = float(inv_las.header.min[1])
    y_max = float(inv_las.header.max[1])

    grid_x, grid_y, grid_z, grid_label = init_cloth(x_min, x_max, y_min, y_max, max(inv_las.z), GR=GR)

    cloth_pts = ClothPoints(grid_x, grid_y, grid_z, grid_label)

    # 3) find the CP for each cloth particle and record intersection height value(== cp_z)
    # spatial indexing for the inversed las
    tree = scipy.spatial.cKDTree(np.column_stack((inv_las.x, inv_las.y)))

    cloth_pts.get_cp_dist_idx(tree)
    cloth_pts.get_cp_z(inv_las.z)

    cloth_pts.reshape()

    # 4) displace the cloth considering the external and internal forces
    row_n = cloth_pts.xs.shape[0]
    col_n = cloth_pts.xs.shape[1]

    zs_prev = cloth_pts.zs.copy()
    total_itr_num = itr_num

    print('start ground filtering')
    while itr_num > 0:
        # displace the cloth
        zs_cur = cloth_pts.zs.copy()

        for i in range(0, row_n):
            for j in range(0, col_n):
                cloth_pts.external_force(i, j, zs_prev[i][j], dT)
                cloth_pts.intersection_check(i, j)

        cloth_pts.internal_force()

        # process termination criteria 1) the maximum height variation of all particles is smaller than the threshold
        # M_HV: maximum height variation(change)
        disp = np.abs(cloth_pts.zs - zs_cur)
        M_HV = np.max(disp)

        if M_HV < max_delta:
            print(f'terminate ground filtering process : M_HV({M_HV}) is less than threshold')
            break

        zs_prev = zs_cur.copy()

        print(f'> iteration #{total_itr_num - itr_num + 1} done')
        
        # process termination criteria 2) it exceeds the maximum iteration number
        itr_num -= 1

        if itr_num == 0:
            print(f'terminate filtering process: maximum iteration number reached')

    return cloth_pts


def ground_classification(cloth_pts, input_file, hcc):
    """
    Function to process ground classification

    Input:
        cloth_pts: displaced cloth
        input_file: original las file
        hcc: distance threshold

    Return:
        org_las: orginal las dataset with ground classification
    """
    # 1) invert the cloth
    inv_cloth_pts = cloth_pts.pts_copy()
    inv_cloth_pts.zs *= -1

    # 2) make a DT from with the inverted cloth points
    inv_cloth_pts_2d = np.column_stack((inv_cloth_pts.xs.ravel(), inv_cloth_pts.ys.ravel(), inv_cloth_pts.zs.ravel()))
    cloth_dt = startinpy.DT()
    cloth_dt.insert(inv_cloth_pts_2d)

    # 3) project original las points to 2d plane
    org_las = laspy.read(input_file)
    org_las_pts_xy = np.column_stack((np.array(org_las.points.x).ravel(), np.array(org_las.points.y).ravel()))

    # 4) get interpolated z values for the projected las points
    intp_zs = cloth_dt.interpolate({"method": "TIN"}, org_las_pts_xy)

    # 5) classify as ground if z difference is less than threshold
    for i in range(0, len(org_las.points.z)):
        diff = abs(intp_zs[i] - org_las.points.z[i])

        if diff < hcc or np.isclose(diff, hcc):
            org_las.classification[i] = 2
        else:
            org_las.classification[i] = 1
          
    return org_las


def output_dtm(classified_las):
    # 1) create an empty grid(1m*1m)
    x_min = float(classified_las.header.min[0])
    x_max = float(classified_las.header.max[0])
    y_min = float(classified_las.header.min[1])
    y_max = float(classified_las.header.max[1])

    xs = np.arange(x_min, x_max + 1, 1)
    ys = np.arange(y_min, y_max + 1, 1)

    num_xs = len(xs)  # num of columns
    num_ys = len(ys)  # num of rows

    grid_x, grid_y = np.meshgrid(xs, ys)
    cents = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # 2) create a Delaunay triangulation with ground points
    ground_mask = (classified_las.classification == 2)
    ground_pts = classified_las.points[ground_mask]

    ground_pts_xs = np.array(ground_pts.x)
    ground_pts_ys = np.array(ground_pts.y)
    ground_pts_zs = np.array(ground_pts.z)
    ground_pts_xyzs = np.column_stack((ground_pts_xs, ground_pts_ys, ground_pts_zs))

    dt = startinpy.DT()
    dt.insert(ground_pts_xyzs)

    # 3) process Laplace interpolation
    interp_z = np.array(dt.interpolate({"method": "Laplace"}, cents)).reshape(grid_x.shape)
    interp_z = np.flipud(interp_z)

    # 4) output the resulting file
    transform = rasterio.transform.from_origin(x_min, y_max, 1, 1)
    
    os.makedirs("data", exist_ok=True)  # ensure folder exists
    
    with rasterio.open(
            "data/dtm.tiff",
            "w",
            driver="GTiff",
            height=num_ys,
            width=num_xs,
            count=1,
            dtype=np.float64,
            crs="EPSG:28992",
            transform=transform,
            nodata=-9999.0,
    ) as dtm:
        dtm.write(interp_z, 1)

    print("dtm.tiff done")


def output_las(input_las):
    """
    Function to output ground points

    Input:
        input_las: original las dataset with ground classification
    """
    # 1) set a header of an output file
    output_header = laspy.LasHeader(point_format=input_las.header.point_format, version=input_las.header.version)
    output_header.scales = input_las.header.scales
    output_header.offset = input_las.header.offsets
    output_header.add_crs(CRS.from_epsg(28992))

    # 2) write the output file
    os.makedirs("data", exist_ok=True)  # ensure folder exists

    with laspy.open('data/out.laz', "w", header=output_header) as output_las:
        pts = input_las.points
        output_las.write_points(pts)
