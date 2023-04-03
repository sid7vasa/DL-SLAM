import carla
import random
import pygame
import numpy as np
import open3d as o3d
import time

from carla_utils import add_npc_vehicles
from pygame_utils import ControlObject
from sensor_utils import initialize_camera, initialize_lidar

def initialize_simulation(synchronous=True, num_vehicles=75):
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    if synchronous:
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    if synchronous:
        traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(0)

    # We will aslo set up the spectator so we can see what we do
    spectator = world.get_spectator()
    vehicles = add_npc_vehicles(world, traffic_manager, max_vehicles=num_vehicles)
    print(f"[INFO] Retrieved {len(vehicles)} vehicles from the blueprint")

    return client, world, traffic_manager, spectator, vehicles



if __name__=="__main__":
    # Initialize the simulation
    client, world, traffic_manager, spectator, vehicles = initialize_simulation(synchronous=True, num_vehicles=75)

    # Randomly select a vehicle to follow with the camera
    ego_vehicle = random.choice(vehicles)
    ego_vehicle.set_autopilot(False)

    # Control this vehicle using Control Object from pygame_utils
    controlObject = ControlObject(ego_vehicle)

    # Initialize Sensors:
    camera, renderObject, image_h, image_w = initialize_camera(world, ego_vehicle)
    lidar_sen = initialize_lidar(world,ego_vehicle)


    # Initialise the display
    pygame.init()
    gameDisplay = pygame.display.set_mode((image_w,image_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # Draw black to the display
    gameDisplay.fill((0,0,0))
    gameDisplay.blit(renderObject.surface, (0,0))
    pygame.display.flip()

    # Game loop
    crashed = False

    while not crashed:
        # Advance the simulation time
        world.tick()
        # Update the display
        gameDisplay.blit(renderObject.surface, (0,0))
        pygame.display.flip()
        # Process the current control state
        controlObject.process_control()
        # Collect key press events
        for event in pygame.event.get():
            # If the window is closed, break the while loop
            if event.type == pygame.QUIT:
                crashed = True

            # Parse effect of key press event on control state
            controlObject.parse_control(event)

    # Stop camera and quit PyGame after exiting game loop
    camera.stop()
    # lidar_sen.stop()
    pygame.quit()

