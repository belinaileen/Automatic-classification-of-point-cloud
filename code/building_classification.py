import logging
import laspy
import numpy as np
import scipy.spatial
import os
from pyproj import CRS
from sklearn.cluster import DBSCAN


# ==============================================================================
# 1. THE GEOMETRIC SIEVE (TREE FILTER)
# ==============================================================================
def geometric_sieve_classifier_curvature(coords, radius=1.5, curvature_threshold=0.1, v_curvature_threshold=0.18,
                                         norm=0.1):
    """
    Filter using curvature score and normal vector.

    Spatial Indexing:
        * kdtree = scipy.spatial.KDTree(coords) [line 23]: Creates a fast lookup structure to find nearby points.
        * kdtree.query_ball_point(coords, r=radius) [line 27]: For every point, it finds all neighbors within a 1.0m or
        1.5m radius.

    Eigenvalue Analysis (SVD):

        * np.linalg.svd(cov) [line 39]: Decomposes the local neighborhood into three principal directions.
        * curvature = l3 / np.sum(s) [line 43]: Calculates how much the points deviate from a flat plane.

    Vector Normal Logic:
        * normal = vh[2, :] [line 47]: Extracts the direction of least variance, which is the surface normal.
        * nz = np.abs(normal[2]) [line 48]: Determines if the surface is horizontal ($n_z \approx 1$) or
        vertical ($n_z \approx 0$).

    """
    logging.info(f"Sieve: Filtering by curvature score and normal (r={radius}m)...")
    kdtree = scipy.spatial.KDTree(coords)
    # Initialize containers for the two types of detected geometries
    valid_indices = []  # For standard flat surfaces (Roofs/Floors)
    Valid_vertical_indices = []  # For vertical structures (Sparse Walls)

    # 1. Neighborhood Search
    # Finds all points within the specified radius of the current point
    all_neighbors = kdtree.query_ball_point(coords, r=radius)

    for i, indices in enumerate(all_neighbors):
        # Minimum Density Check: We need at least 5 points to perform
        # reliable Singular Value Decomposition (SVD)
        if len(indices) < 5:
            continue

        # 2. Local Geometry Analysis
        # Center the neighbors around their mean to calculate dispersion
        neighbors = coords[indices]
        centered = neighbors - np.mean(neighbors, axis=0)

        # Generate the 3x3 Covariance Matrix (Scatter Matrix)
        # This represents how points are spread in X, Y, and Z
        cov = np.dot(centered.T, centered)

        # Singular Value Decomposition (SVD)
        # s: Eigenvalues (magnitudes of spread) Size of the "Ellipsoid"
        # vh: Eigenvectors (directions of spread) Orientation of the "Ellipsoid"
        _, s, vh = np.linalg.svd(cov)
        l1, l2, l3 = s[0], s[1], s[2]

        s, v = np.linalg.eigh(cov)

        # 3. Surface Variation (Curvature) Calculation
        # Measures how 'thick' the point cloud is in its thinnest dimension
        curvature = l3 / np.sum(s) if np.sum(s) > 0 else 1.0

        # 4. Normal Vector Extraction
        # The direction of least variance (vh[2]) is the Surface Normal
        # normal = vh[2, :]
        # nz = np.abs(normal[2])  # Vertical component magnitude (0 to 1)

        normal = v[:, 0]
        nz = np.abs(normal[2])

        # 5. Classification Logic
        # Case A: Standard Flat Surface (likely Roofs or Floors)
        # Requires very low curvature (very flat)
        if curvature < curvature_threshold:
            valid_indices.append(i)

        # Case B: Vertical Sparse Structure (specifically for Walls)
        # Detected if the normal is nearly horizontal (nz < 0.2)
        # We allow a higher curvature (0.17) to account for sensor noise in sparse data
        #  and curvature < 0.17
        if nz < norm and curvature < v_curvature_threshold:
            Valid_vertical_indices.append(i)

    return np.array(valid_indices, dtype=np.int64), np.array(Valid_vertical_indices, dtype=np.int64)


# ==============================================================================
# 2. PLANE EXTRACCIÓN (RETURNS A LIST OF LISTS WITH POINTS INDEXS)
# ==============================================================================
def extract_plane_groups(las_obj, sieve_indices, epsilon=0.3, min_points=50,
                         cluster_eps=2.0, iterations=1000, use_clustering=True):
    """
    Extrac planes using RANSAC.

    Data Preparation
        * coords = las_obj.xyz[sieve_indices]:
        This creates a sub-cloud containing only the $(x, y, z)$ coordinates of the points that survived the curvature filter.

        * available_mask = np.ones(len(coords), dtype=bool):
        This is a "checklist" initialized to True. It keeps track of which points have already been assigned to a
        building so they aren't processed twice.

    The RANSAC Loop

    The while loop continues as long as there are enough "available" points to potentially form a valid plane.

        * sample = np.random.choice(active, 3, replace=False): It randomly selects 3 points. In geometry, 3 points are the
        minimum required to define a unique plane in 3D space.

        * np.cross(v1, v2): This performs a Cross Product. The syntax calculates a vector that is perpendicular to the two
        vectors formed by the 3 sampled points; this resulting vector is the Normal Vector of the proposed plane.

        * dists = np.abs(np.dot(coords[active], normal) + d): This line applies the general plane equation
        ($Ax + By + Cz + D = 0$) to all points simultaneously using matrix algebra.

        * inliers = active[dists < epsilon]: This filters and stores the indices of points whose distance to the mathematical
        plane is smaller than the epsilon (e.g., 0.3 meters).


    """
    coords = las_obj.xyz[sieve_indices]
    available_mask = np.ones(len(coords), dtype=bool)
    plane_groups = []

    logging.info(f"RANSAC: Searching planes (Clustering={'ON' if use_clustering else 'OFF'}) en {len(coords)} points...")

    while np.sum(available_mask) > min_points:
        active = np.where(available_mask)[0]
        if len(active) < 3: break

        best_s = 0
        best_inliers = None

        # 1. RANSAC:
        for _ in range(iterations):
            sample = np.random.choice(active, 3, replace=False)
            pts = coords[sample]
            v1, v2 = pts[1] - pts[0], pts[2] - pts[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm == 0: continue
            normal /= norm
            d = -np.dot(normal, pts[0])

            dists = np.abs(np.dot(coords[active], normal) + d)
            inliers = active[dists < epsilon]

            if len(inliers) > best_s:
                best_s = len(inliers)
                best_inliers = inliers

        """
        Spatial Refinement (DBSCAN)

            * if use_clustering:: 
            If set to True, the code attempts to separate the points found by RANSAC based on their physical proximity.

            * DBSCAN(eps=cluster_eps, min_samples=10).fit(...):
                a - eps=cluster_eps (2.0m): Defines the maximum distance for two points to be considered "neighbors".    
                b - labels = clustering.labels_: Assigns an ID to each found group. Isolated points that don't belong 
                to a dense group receive the ID -1 (noise).

            * if len(actual_inliers) >= min_points: 
            We only save the group if it has enough points to be considered a real object (e.g., a full wall rather than 
            just a stray cluster of 5 points).
            """

        if best_inliers is not None and len(best_inliers) >= min_points:

            # --- SWITCH TO TURN OFF THE DBSCAN ---
            if use_clustering:
                # 2. DBSCAN:
                clustering = DBSCAN(eps=cluster_eps, min_samples=10).fit(coords[best_inliers])
                labels = clustering.labels_

                found_valid_cluster = False
                for cluster_id in np.unique(labels):
                    if cluster_id == -1: continue

                    """
                    Cleanup and Updating

                        * available_mask[actual_inliers] = False: 
                        This uses NumPy boolean indexing. it marks the points of the detected plane as "used" so the while 
                        loop ignores them in the next iteration.

                        * plane_groups.append(...): 
                        It adds the validated group of points to the final list, which is eventually exported as an 
                        individual .laz file.
                    """

                    cluster_mask = (labels == cluster_id)
                    actual_inliers = best_inliers[cluster_mask]

                    if len(actual_inliers) >= min_points:
                        plane_groups.append(sieve_indices[actual_inliers])
                        available_mask[actual_inliers] = False
                        found_valid_cluster = True
                        logging.info(f"  -> Plane #{len(plane_groups)} (Cluster): {len(actual_inliers)} points.")

                # Avoid bucles if RANSAC founds something that DBSCAN rejects
                if not found_valid_cluster:
                    available_mask[best_inliers] = False

            else:
                # 2. WITHOUT CLUSTERING: Just export plane index
                plane_groups.append(sieve_indices[best_inliers])
                available_mask[best_inliers] = False
                logging.info(f"  -> Plane #{len(plane_groups)} (RANSAC Puro): {len(best_inliers)} points.")
        else:
            break

    return plane_groups


# ==============================================================================
# 3. IMPLEMENTATION
# ==============================================================================
def building_classification(input_file,
                            radius=2.319491227584038,
                            h_curv_threshold=0.14871608260734354,
                            v_curv_threshold=0.12176416804489787,
                            v_norm = 0.05954323977762549,
                            h_ransac_eps = 0.5521997565193092,
                            v_ransac_eps = 0.3620889061367161,
                            h_min_pts = 26,
                            v_min_pts = 19,
                            h_cluster_eps = 1.765793902126422,
                            v_cluster_eps = 1.048552922121527):

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    las_original = laspy.read(input_file)

    mask = (las_original.classification == 1)
    las = las_original[mask]

    # 1.  Sieve
    v_horiz, v_vert = geometric_sieve_classifier_curvature(
        las.xyz, radius=radius, curvature_threshold=h_curv_threshold, v_curvature_threshold=v_curv_threshold,
        norm=v_norm
    )
    # 2. Extraction of planes
    horizontal_tilted_plane_list = extract_plane_groups(las, v_horiz, epsilon=h_ransac_eps, min_points=h_min_pts,
                                                        cluster_eps=h_cluster_eps)
    vertical_plane_list = extract_plane_groups(las, v_vert, epsilon=v_ransac_eps, min_points=v_min_pts,
                                               cluster_eps=v_cluster_eps)
    all_planes = horizontal_tilted_plane_list + vertical_plane_list

    # 3. Update the las classification values and export

    # a. Apply found classification
    for plane in all_planes:
        for point in plane:
            las.classification[point] = 6

    # b. Build laz header
    output_header = laspy.LasHeader(point_format=las_original.header.point_format, version=las_original.header.version)
    output_header.scales = las_original.header.scales
    output_header.offset = las_original.header.offsets
    output_header.add_crs(CRS.from_epsg(28992))

    # c. Write file
    os.makedirs("data", exist_ok=True)  # ensure folder exists
    with laspy.open('data/out.laz', "w", header=output_header) as output_las:
        ground_mask = (las_original.classification == 2)
        ground_pts = las_original.points[ground_mask]
        output_las.write_points(ground_pts)

        tree_mask = (las_original.classification == 5)
        tree_pts = las_original.points[tree_mask]
        output_las.write_points(tree_pts)

        building_mask = (las.classification == 6)
        building_pts = las.points[building_mask]
        output_las.write_points(building_pts)

        non_classified_mask = (las.classification == 1)
        non_classified_pts = las.points[non_classified_mask]
        output_las.write_points(non_classified_pts)

