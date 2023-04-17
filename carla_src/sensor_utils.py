import carla
import numpy as np
import open3d as o3d

from pygame_utils import RenderObject, pygame_callback

def initialize_camera(config_dict, world, ego_vehicle):
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
    camera.listen(lambda image: pygame_callback(config_dict, image, renderObject, world.get_snapshot().frame))
    return camera, renderObject, image_h, image_w

def initialize_lidar(config_dict, world, ego_vehicle):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels',str(64))
    lidar_bp.set_attribute('points_per_second',str(4500000))
    lidar_bp.set_attribute('range',str(120))
    lidar_bp.set_attribute('rotation_frequency', str(30))
    lidar_location = carla.Location(0,0,2)
    lidar_rotation = carla.Rotation(0,0,0)
    lidar_transform = carla.Transform(lidar_location,lidar_rotation)
    lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle)
    
    path = config_dict['data_collection']['path']
    gps_data = []
    lidar_data = []
    transform_data = []

    def process_lidar(point_cloud):
        # Get the current GPS location and rotation of the vehicle
        vehicle_transform = ego_vehicle.get_transform()
        transform_data.append({
            'location': {
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z
            },
            'rotation': {
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw,
                'roll': vehicle_transform.rotation.roll
            }
        })

        # Process the LIDAR data and add it to the buffer
        lidar_data.append(point_cloud)

        # Save the GPS, LIDAR, and transform data to disk
        point_cloud.save_to_disk(f'{path}lidar/{point_cloud.frame}.ply')
        np.save(f'{path}gps/{point_cloud.frame}.npy', transform_data[-1])

    lidar_sen.listen(process_lidar)

    return lidar_sen
