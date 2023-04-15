import numpy as np
import torch
import open3d as o3d
from typing import List

class DCP:
    def __init__(self, model_path, num_down_sample_points=1024):
        self.net = torch.load(model_path)
        self.num_down_sample_points = num_down_sample_points

    def downsample_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.PointCloud]:
        downsampled_clouds = []
        for pcd in point_clouds:
            # Calculate the sampling ratio required to downsample to 1024 points
            num_points = len(pcd.points)
            sampling_ratio = 1024 / num_points
            # Downsample the point cloud to 1024 points using random sampling
            downsampled_pcd = pcd.random_down_sample(sampling_ratio)
            downsampled_pcd_points = np.asarray(downsampled_pcd.points).T.astype(np.float32)
            # Convert to PyTorch tensor of shape (3, 1024)
            downsampled_pcd_points_tensor = torch.from_numpy(downsampled_pcd_points).reshape(3, 1024)
            downsampled_clouds.append(downsampled_pcd_points_tensor)
        return downsampled_clouds

    def infer(self, source_path, target_path):
        # Load the list of point clouds from the .ply files
        point_clouds = [o3d.io.read_point_cloud(source_path), o3d.io.read_point_cloud(target_path)]
        # Downsample the list of point clouds to 1024 points each
        downsampled_clouds = self.downsample_point_clouds(point_clouds)
        print("DOWN SAMPLED POINTS:",[pcd.shape for pcd in downsampled_clouds])
        self.net.eval()
        source = downsampled_clouds[0].unsqueeze(0).to("cuda")
        target = downsampled_clouds[1].unsqueeze(0).to("cuda")
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = self.net.forward(source, target)
        return rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred
    

if __name__ == "__main__":
    # Source and target point clouds to perform DCP
    source_path = "/home/sid/lidar/000262.ply"
    target_path = "/home/sid/lidar/000278.ply"

    # Loading the network
    model_path = 'pretrained/full_modelv1.pt'
    dcp = DCP(model_path, num_down_sample_points=1024)
    rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = dcp.infer(source_path, target_path)
    print(rotation_ab_pred.shape)
    print(translation_ab_pred.shape)
    print(rotation_ba_pred.shape)
    print(translation_ba_pred.shape)
    