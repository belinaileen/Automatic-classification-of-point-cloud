import laspy
import numpy as np
from scipy.spatial import cKDTree
import os


def trees_detection(las_file):
    """
    Stacking the points as (x,y,z) points as an array.
    .T -> So as to transponse the dataset from (3,N) to (N,3) format. ( For KD-Tree)
    """
    las = laspy.read(las_file)
    points=np.vstack((las.x,las.y,las.z)).T
    classifications=las.classification
    num_returns = las.number_of_returns
    unique_classes, counts = np.unique(np.asarray(classifications), return_counts=True)

    """
    KD Tree Structure 

    It's a Binary tree which splits the space based along (x,y,z)

    Build: O(nlogn)
    Query : O(logn)

    cKDTree -> C++ KD Tree (Faster Execution)

    """
    tree=cKDTree(points)

    """
    Making a Boolean array of the size of the points available and labelling only the 
    points which satisfy as a tree point as TRUE.
    """
    veg_mask=np.zeros(len(points),dtype=bool)


    #Looping over KD Tree.
    """
    Automating the value for k and z_min_tree based on the AHN tile given.

    1. K (number of neighbors for PCA)
        If they are too small -> noisy PCA calculation
        If Too large -> Chances of wrong classification due to smoothing of structure.

        Approach: 
        Compute average spacing between points (point density)
        Choose k so that neighbors fall within a small radius around the point.
    

    2. z_min_tree (minimum tree height)
        Compute z-percentile and ignore the lowest 5% of points.

    """
    k=64
    #Minimum height of the Tree for consideration
    z_min_tree=np.percentile(points[:, 2],5)

    #Classifying Tree Crowns
    for i,pt in enumerate(points):
        #Skip points that are already classified
        if las.classification[i]!=0 and las.classification[i]!=1:
            continue

        #Skip points that are too low - (Small plants and Shrubs)
        if pt[2]<z_min_tree:
            continue

        if num_returns[i]<=1:
            continue

        #Find k nearest neighbours
        """
        Finding the closest 20 points (k=k) for point pt.
        And also their respective distances to the points.

        First point is the point itself
        """
        dists, idx=tree.query(pt,k=k)
        neighbors=points[idx]

        #Computing covariance matrix

        """
        Computing local shape analysis using PCA(Principal Component Analysis)
        To measure what the neighbouring point forms (Line, Plane, Volume)
    
        Firstly the covariance of the points are calculated in a covariance matrix. 
        (3,N) format needed by Numpy.

        Helps in understanding ho tightly/loosely clustered the points are.

        Upon this, the eigenvalues is calculated.
        Eigenvalues represent variance along principal directions.
        det(A-lamba*I)=0 returns a cubic equation which gives the eigenvalues
        """

        if neighbors.shape[0] < 2:
            continue

        cov=np.cov(neighbors.T)
        eigvals,eigvecs=np.linalg.eigh(cov)

        #Computing Planarity and Sphericity
        """
        Calculating plane or volumetric characteristic of the k nearest points resulted to be.
        Basically calculating the level of spread.
        Planarity -> How the points have spread on the plane. 
        Sphericity -> How the points have spread in volumetric setting. 
        """
        planarity=(eigvals[1]-eigvals[0])/eigvals[2]
        sphericity=eigvals[0]/eigvals[2]
        
        #Mark point as tree if it matches criteria.
        if planarity<0.7 and sphericity>0.05:
            veg_mask[i]=True
        """
        Detecting Tree Trunks
        """
    las.classification[veg_mask] = 5
    print("Status: Tree classification Complete")

    os.makedirs("data", exist_ok=True)  # ensure folder exists
    output_file='data/out.laz'
    las.write(output_file)

#if __name__ == "__main__":
 #   trees_detection("data/AHN Cropped/AHN_Cropped.laz")
