import carla
import numpy as np
import open3d as o3d

def initialize_camera(world, ego_vehicle):
    # Initialise the camera floating behind the vehicle
    camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    return camera, camera_bp

def initialize_lidar(world, ego_vehicle):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',str(32))
    lidar_bp.set_attribute('points_per_second',str(90000))
    lidar_bp.set_attribute('range',str(20))
    lidar_location = carla.Location(0,0,2)
    lidar_rotation = carla.Rotation(0,0,0)
    lidar_transform = carla.Transform(lidar_location,lidar_rotation)
    lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle)
    lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('/home/carla/PythonAPI/workspace/DL-SLAM/data/lidar/%.6d.pcd' % point_cloud.frame))
    return lidar_sen