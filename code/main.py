# -- geo1015.2025.hw03

import argparse
import sys
import laspy

import preprocess
import ground_classification
import building_classification
import tree_classification

def main():
    parser = argparse.ArgumentParser(description="My GEO1015.2025 hw03")

    parser.add_argument("inputfile", help="Input file (required)")

    # csf argument
    parser.add_argument("--csf_r", type=float, default=2.32, help="CSF grid resolution")
    parser.add_argument("--csf_dT", type=float, default=0.64, help="time step for the external force")
    parser.add_argument("--csf_e", type=float, default=1.27, help="CSF espilson")
    parser.add_argument(
        "--csf_zmax",
        type=float,
        default=0.0001,
        help="CSF tolerance espilon_zmax to stop the iterations",
    )
    parser.add_argument("--csf_it", type=int, default=500, help="CSF max iternation number")

    args = parser.parse_args()

    try:
        lazfile = laspy.open(args.inputfile)
        print(f'{args.inputfile} ready')
    except Exception as e:
        print(e)
        sys.exit()

    gr = args.csf_r
    dT = args.csf_dT
    hcc = args.csf_e
    max_delta = args.csf_zmax
    itr_num = args.csf_it

    print(
        f'<Parameter setup>\ncloth resolution {gr}\ndT {dT}\nhcc {hcc}\nmax_delta {max_delta}\nmax iteration number {itr_num}\n')

    # 1) preprocess
    las_path = preprocess.preprocess(lazfile)

    # 2) ground filtering
    displaced_cloth = ground_classification.cloth_displacement(las_path, itr_num, gr, dT, max_delta)
    g_classified_las = ground_classification.ground_classification(displaced_cloth, las_path, hcc)
    ground_classification.output_dtm(g_classified_las)
    ground_classification.output_las(g_classified_las)

    # 3) tree filtering
    tree_classification.trees_detection('data/out.laz')

    # 4) building filtering
    building_classification.building_classification('data/out.laz')


if __name__ == "__main__":
    main()
