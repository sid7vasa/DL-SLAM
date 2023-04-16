import argparse
import os
import open3d as o3d
import UtilsPointcloud as Ptutils
# Example command to run this script:
# python3 bin_to_ply.py --input /home/sid/workspace/rss/data/data_odometry_velodyne/dataset/sequences/00/velodyne/ --output /home/sid/lidar_kitti/00

def convert_bin_to_ply(scan_directory_path, write_directory_path):
    """
    Converts a directory of bin files containing point clouds to PLY files using Open3D.

    Args:
    - scan_directory_path (str): Path to the directory containing the bin files.
    - write_directory_path (str): Path to the directory where the PLY files will be written.

    Returns:
    - None
    """
    # Make sure the write directory exists
    if not os.path.exists(write_directory_path):
        os.makedirs(write_directory_path)

    # Loop over all files in the scan directory
    for filename in os.listdir(scan_directory_path):
        # Check that the file is a bin file
        if not filename.endswith('.bin'):
            continue

        # Read the point cloud from the bin file
        scan_path = os.path.join(scan_directory_path, filename)
        points = Ptutils.readScan(scan_path)

        # Convert the point cloud to an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Write the point cloud to a PLY file
        write_path = os.path.join(write_directory_path, f'{filename[:-4]}.ply')
        o3d.io.write_point_cloud(write_path, pcd)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description='Convert bin files to PLY files.')
    parser.add_argument('--input', type=str, required=True, help='Path to directory containing the input bin files.')
    parser.add_argument('--output', type=str, required=True, help='Path to directory where the output PLY files will be written.')

    # Parse arguments
    args = parser.parse_args()

    # Run the conversion function
    convert_bin_to_ply(args.input, args.output)
