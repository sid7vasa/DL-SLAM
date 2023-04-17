import random
import numpy as np

def add_npc_vehicles(world, traffic_manager, max_vehicles=50, seed=0):
    # random.seed(seed)

    # Retrieve the map's spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Select some models from the blueprint library
    blueprints = []
    for vehicle in world.get_blueprint_library().filter('*vehicle*'):
        blueprints.append(vehicle)

    # Set a max number of vehicles and prepare a list for those we spawn
    max_vehicles = min([max_vehicles, len(spawn_points)])
    vehicles = []

    # Take a random sample of the spawn points and spawn some vehicles
    for i, spawn_point in enumerate(random.sample(spawn_points, max_vehicles)):
        temp = world.try_spawn_actor(random.choice(blueprints), spawn_point)
        if temp is not None:
            vehicles.append(temp)

    # Parse the list of spawned vehicles and give control to the TM through set_autopilot()
    for vehicle in vehicles:
        vehicle.set_autopilot(True)
        # Randomly set the probability that a vehicle will ignore traffic lights
        traffic_manager.ignore_lights_percentage(vehicle, random.randint(0,50))

    return vehicles

def get_homogeneous_matrix(pose1, pose2):
    # Extract the position and orientation parameters for frames 1 and 2
    x1, y1, z1, roll1, yaw1, pitch1 = pose1
    x2, y2, z2, roll2, yaw2, pitch2 = pose2
    
    # Compute the rotation matrices for each frame
    R1 = get_rotation_matrix(roll1, yaw1, pitch1)
    R2 = get_rotation_matrix(roll2, yaw2, pitch2)
    
    # Compute the translation vector
    t = np.array([[x2 - x1], [y2 - y1], [z2 - z1]])
    
    # Combine the rotation and translation matrices into a single homogeneous matrix
    R12 = R2 @ R1.T
    t12 = -R2 @ R1.T @ t
    H12 = np.vstack((np.hstack((R12, t12)), np.array([[0, 0, 0, 1]])))
    
    return H12

def get_rotation_matrix(roll, yaw, pitch):
    # Compute the rotation matrix from roll, yaw, and pitch angles
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(pitch), -np.sin(pitch), 0], [np.sin(pitch), np.cos(pitch), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    return R

