import os
import pandas as pd
import numpy as np
import argparse

from carla_utils import get_homogeneous_matrix

def load_data(data_dir):
    # Load the GPS data into a pandas dataframe
    gps_files = os.listdir(os.path.join(data_dir, "gps"))
    gps_data = []
    for file in gps_files:
        if file.endswith(".npy"):
            file_path = os.path.join(data_dir, "gps", file)
            gps = np.load(file_path, allow_pickle=True).item()
            location = gps['location']
            rotation = gps['rotation']
            gps_data.append([file[:-4], location['x'], location['y'], location['z'], rotation['pitch'], rotation['yaw'], rotation['roll']])
    gps_df = pd.DataFrame(gps_data, columns=['file', 'x', 'y', 'z', 'pitch', 'yaw', 'roll'])

    # Load the LiDAR data into a list
    lidar_files = os.listdir(os.path.join(data_dir, "lidar"))
    lidar_data = [file[:-4] for file in lidar_files if file.endswith(".ply")]

    # Create a pandas dataframe with the LiDAR file names
    lidar_df = pd.DataFrame(lidar_data, columns=['file'])

    # Merge the GPS and LiDAR dataframes on the 'file' column
    merged_df = pd.merge(gps_df, lidar_df, on='file')

    # Convert the 'file' column to integers, sort the dataframe, and reset the index
    merged_df['file'] = merged_df['file'].astype(int)
    merged_df = merged_df.sort_values(by='file').reset_index(drop=True)

    return merged_df

def create_consecutive_pairs_df(merged_df):
    # Create a list of dictionaries for consecutive file pairs
    pairs_list = []
    for i in range(1, len(merged_df)):
        # Extract the file names from the current and previous rows
        source_file = merged_df.iloc[i]['file']
        target_file = merged_df.iloc[i-1]['file']
        # Add the file names to the pairs list as a dictionary
        pairs_list.append({'source': source_file, 'target': target_file})

    # Create the consecutive pairs dataframe from the pairs list
    consecutive_pairs = pd.DataFrame(pairs_list)

    return consecutive_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Add a command-line argument for the data directory path
    parser.add_argument("data_dir", type=str, help="path to the data directory", nargs="?", default="/home/sid/scans/DATA_2023-04-16_23-36-03")

    # Add a command-line argument for the output file path
    parser.add_argument("write_path", type=str, help="path to the output CSV file", nargs="?", default="transform_tables/consecutive_pairs")


    args = parser.parse_args()

    time_stamp = args.data_dir.split("/")[-1]
    if time_stamp == "":
        time_stamp = args.data_dir.split("/")[-2]

    print("TIME_STAMP:", time_stamp)
    args.write_path = args.write_path + "_" + time_stamp + ".csv"

    # Define the path to the data directory
    data_dir = args.data_dir

    # Load the data and create the consecutive pairs dataframe
    merged_df = load_data(data_dir)
    consecutive_pairs = create_consecutive_pairs_df(merged_df)

    # Compute the homogeneous matrices for each consecutive pair
    homogeneous_matrices = []
    for i, row in consecutive_pairs.iterrows():
        # Extract the pose information for the source and target files
        source_pose = merged_df.loc[merged_df['file'] == row['source']].values[0][1:]
        target_pose = merged_df.loc[merged_df['file'] == row['target']].values[0][1:]
        # Call the get_homogeneous_matrix() function with the extracted pose information
        homogeneous_matrix = get_homogeneous_matrix(source_pose, target_pose)
        homogeneous_matrices.append(homogeneous_matrix)

    # Add the homogeneous matrices to the consecutive pairs dataframe
    consecutive_pairs['homogeneous_matrix'] = homogeneous_matrices
    
    # Add the parent folder and ".ply" suffix to the file names
    print(consecutive_pairs["source"].astype(str))
    consecutive_pairs["source"] = data_dir.split("/")[-1] + "/lidar/" + consecutive_pairs["source"].astype(int).astype(str) + ".ply"
    consecutive_pairs["target"] = data_dir.split("/")[-1] + "/lidar/" + consecutive_pairs["target"].astype(int).astype(str) + ".ply"


    consecutive_pairs.to_csv(args.write_path, index=False)

    # Display the consecutive pairs dataframe
    print(consecutive_pairs.head())