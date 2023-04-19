import torch
import numpy as np
import open3d as o3d
from transforms3d import affines
from pcrnet.models import PointNet
from pcrnet.models import iPCRNet


class PointCloudRegistrationModel:
    def __init__(self, model_path):
        # Create an instance of the feature model (PointNet)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ptnet = PointNet(emb_dims=1024)

        # Create an instance of the iPCRNet model
        self.model = iPCRNet(feature_model=ptnet)

        self.model = self.model.to(self.device)

        # Load the state dictionaries
        model_snap = torch.load(model_path)
        self.model.load_state_dict(model_snap['model'])
        print("[INFO] Loaded PCR model Successfully")
 


    def downsample(self, source, target, num_points):
        source_rate = int(len(source.points) / num_points)
        target_rate = int(len(target.points) / num_points)


        # Downsample the point clouds
        source_pcd_down = source.uniform_down_sample(source_rate)
        target_pcd_down = target.uniform_down_sample(target_rate)

        if len(source_pcd_down.points) > num_points:
            source_pcd_down.points = source_pcd_down.points[:num_points]

        if len(target_pcd_down.points) > num_points:
            target_pcd_down.points = target_pcd_down.points[:num_points]
        
        return source_pcd_down, target_pcd_down


    def predict_transformation(self, source_points, target_points):
        """
        Function that takes source and target point clouds (in o3d pointcloud format or numpy array)
        and returns the 4x4 transformation matrix outputted by the network.

        :param source_points: source point cloud, in o3d pointcloud format or numpy array
        :param target_points: target point cloud, in o3d pointcloud format or numpy array
        :return: 4x4 transformation matrix
        """
        source_points, target_points = self.downsample(source_points, target_points, num_points=2048)

        # Convert to numpy arrays if necessary
        if isinstance(source_points, o3d.geometry.PointCloud):
            source_points = np.asarray(source_points.points)
        if isinstance(target_points, o3d.geometry.PointCloud):
            target_points = np.asarray(target_points.points)

        # Normalize the points
        source_points -= source_points.mean(axis=0)
        target_points -= target_points.mean(axis=0)

        source_range = np.max(np.abs(source_points), axis=0)
        target_range = np.max(np.abs(target_points), axis=0)

        source_points /= np.max(source_range)
        target_points /= np.max(target_range)

        # Convert to PyTorch tensors
        source_points = torch.from_numpy(source_points).type(torch.FloatTensor).unsqueeze(0)
        target_points = torch.from_numpy(target_points).type(torch.FloatTensor).unsqueeze(0)

        source_points = source_points.to(self.device)
        target_points = target_points.to(self.device)


        # Predict the transformation matrix
        result = self.model(source_points, target_points)
        est_R = result['est_R'][0].detach().cpu().numpy()
        est_t = result['est_t'][0].detach().cpu().numpy()
        est_T = result['est_T'][0].detach().cpu().numpy()
        return est_T

if __name__ == "__main__":
    # Example Usage:
    checkpoint_path = "/home/sid/workspace/rss/DL-SLAM/checkpoints/best_model_snap.t7"
    target = "/home/sid/scans/Good/DATA_2023-04-17_17-27-07/lidar/88.ply"
    source = "/home/sid/scans/Good/DATA_2023-04-17_17-27-07/lidar/98.ply"

    model = PointCloudRegistrationModel(checkpoint_path)
    target = o3d.io.read_point_cloud(target)
    source = o3d.io.read_point_cloud(source)
    result = model.predict_transformation(source, target)
    print(result)