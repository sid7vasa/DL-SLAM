import numpy as np
import open3d as o3d
import cv2
import os
import re

def point_cloud_to_image(pcd):
    """
    Convert an Open3D point cloud to an image numpy array.
    
    Parameters:
    -----------
    pcd : open3d.geometry.PointCloud
        Input point cloud.
    
    Returns:
    --------
    img : numpy.ndarray
        Output image numpy array.
    """
    # Convert point cloud to numpy array.
    points = np.asarray(pcd.points)
    
    # Normalize points.
    points -= np.min(points, axis=0)-1
    points /= np.max(points, axis=0)+1
    
    # Rescale points to image size.
    w, h = 640, 480
    points[:, 0] *= w
    points[:, 1] *= h
    
    # Create empty image.
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Project points onto image.
    for i in range(points.shape[0]):
        x, y = int(points[i, 0]), int(points[i, 1])
        img[y, x, :] = (255, 255, 255)
    
    return img

def convert_sequence_to_img_list(ply_files):
    """
    Convert a sequence of point cloud files to a list of images.
    
    Parameters:
    -----------
    ply_files : list of str
        List of input PLY file paths.
    
    Returns:
    --------
    img_list : list of numpy.ndarray
        List of output images.
    """
    img_list = []
    for file_path in ply_files:
        # Load point cloud from file.
        pcd = o3d.io.read_point_cloud(file_path)

        # Convert point cloud to image.
        img = point_cloud_to_image(pcd)

        # Append image to list.
        img_list.append(img)
    
    return img_list

def images_to_video(images, video_path, fps=25):
    """
    Convert a list of images to a video using OpenCV.
    
    Parameters:
    -----------
    images : list of numpy.ndarray
        List of input images.
    video_path : str
        Output video file path.
    fps : int, optional
        Frames per second for the output video. Default is 25.
    """
    # Get image size from first image in the list.
    h, w, c = images[0].shape
    
    # Create video writer object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 format.
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    # Loop over images and write them to video.
    for img in images:
        out.write(img)
    
    # Release video writer object and close all windows.
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # Set directory path
    dir_path = "/home/sid/scans/DATA_2023-04-16_23-36-03/lidar"

    # Get list of all PLY files in directory
    ply_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.ply')]
    pattern = re.compile(r'\d+')
    ply_files = sorted(ply_files, key=lambda path: int(pattern.findall(path)[-1]))
    print(f"Found {len(ply_files)} Point Clouds")

    # convert all the ply files into images to write to a video later.
    img_list = convert_sequence_to_img_list(ply_files)

    # Write image list as a video file
    images_to_video(img_list, "output.mp4", fps=30)

