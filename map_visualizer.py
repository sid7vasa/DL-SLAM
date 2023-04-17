import argparse
import numpy as np
import open3d as o3d
import pandas as pd
import os
import re

# Example command to run this file:
# python3 map_visualizer.py /home/sid/lidar_kitti/00 /home/sid/workspace/rss/DL-SLAM/PyICP-SLAM/result/00/pose00optimized_1681608043.csv map_00.ply

# Set up the argument parser
parser = argparse.ArgumentParser(description='Create a point cloud map from a sequence of point clouds and transformation matrices.')
parser.add_argument('input_dir', type=str, help='The directory containing the input point cloud files.')
parser.add_argument('csv_file', type=str, help='The CSV file containing the transformation matrices.')
parser.add_argument('output_file', type=str, help='The output file name for the map point cloud.')
parser.add_argument('--sampling_ratio', type=float, default=0.005, help='The ratio of points to sample from each point cloud.')

# Parse the command line arguments
args = parser.parse_args()

# Load the CSV file into a Pandas dataframe
df = pd.read_csv(args.csv_file, header=None)

# Extract the transformation matrices from the dataframe as Numpy arrays
T_list = []
for i in range(len(df)):
    row_values = df.iloc[i, :].values
    matrix = np.reshape(row_values, (4, 4)).astype('float64')
    T_list.append(matrix)

# Load the point cloud scans and apply the transformation matrices
pcd_list = []
scan_paths = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir)]
scan_paths = sorted(scan_paths, key=lambda path: int(path.split("/")[-1][:3]))
for i in range(len(T_list)):
    pcd = o3d.io.read_point_cloud(scan_paths[i])
    pcd.transform(T_list[i])
    pcd_down = pcd.random_down_sample(sampling_ratio=args.sampling_ratio)
    pcd_list.append(pcd_down)

# Create the map by combining all point clouds
map_pcd = pcd_list[0]
for i in range(1, len(pcd_list)):
    map_pcd += pcd_list[i]

# Save the down-sampled map as a PLY file
o3d.io.write_point_cloud(args.output_file, map_pcd)

# Visualize the map
o3d.visualization.draw_geometries([map_pcd])
