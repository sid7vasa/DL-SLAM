import carla
import random
import pygame
import numpy as np
import open3d as o3d
import time

from carla_utils import add_npc_vehicles
from pygame_utils import ControlObject, RenderObject, pygame_callback


if __name__=="__main__":
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(0)

    # We will aslo set up the spectator so we can see what we do
    spectator = world.get_spectator()

    vehicles = add_npc_vehicles(world, traffic_manager, max_vehicles=75)

    # Randomly select a vehicle to follow with the camera
    ego_vehicle = random.choice(vehicles)
    ego_vehicle.set_autopilot(False)

    # Initialise the camera floating behind the vehicle
    camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1920')
    camera_bp.set_attribute('image_size_y', '1080')
    camera_bp.set_attribute('fov', '110')
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

    # Start camera with PyGame callback
    camera.listen(lambda image: pygame_callback(image, renderObject))

    # Get camera dimensions
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()

    # Instantiate objects for rendering and vehicle control
    renderObject = RenderObject(image_w, image_h)
    controlObject = ControlObject(ego_vehicle)

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
            print(event)
            # If the window is closed, break the while loop
            if event.type == pygame.QUIT:
                crashed = True

            # Parse effect of key press event on control state
            controlObject.parse_control(event)
            if event.type == pygame.KEYUP:
                # TAB key switches vehicle
                if event.key == pygame.K_TAB:
                    ego_vehicle.set_autopilot(True)
                    ego_vehicle = random.choice(vehicles)
                    ego_vehicle.set_autopilot(False)

                    # Ensure vehicle is still alive (might have been destroyed)
                    if ego_vehicle.is_alive:
                        # Stop and remove the camera
                        camera.stop()
                        camera.destroy()

                        # Spawn new camera and attach to new vehicle
                        controlObject = ControlObject(ego_vehicle)
                        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
                        camera.listen(lambda image: pygame_callback(image, renderObject))

                        # Update PyGame window
                        gameDisplay.fill((0,0,0))               
                        gameDisplay.blit(renderObject.surface, (0,0))
                        pygame.display.flip()

    # Stop camera and quit PyGame after exiting game loop
    camera.stop()
    pygame.quit()


