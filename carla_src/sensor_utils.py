import carla
import numpy as np
import open3d as o3d

from pygame_utils import RenderObject, pygame_callback

def initialize_camera(world, ego_vehicle):
    """
    This function also returns renger object as camera sensor also has to display a window.
    This window has to be rendered using RenderObject and has image height and width assoiciated to it. 
    """
    # Initialise the camera floating behind the vehicle
    camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    # Get camera dimensions
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    # Instantiate objects for rendering and vehicle control
    renderObject = RenderObject(image_w, image_h)
    # Start camera with PyGame callback
    camera.listen(lambda image: pygame_callback(image, renderObject))
    return camera, renderObject, image_h, image_w

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