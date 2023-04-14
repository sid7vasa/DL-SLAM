import random

def add_npc_vehicles(world, traffic_manager, max_vehicles=50, seed=0):
    random.seed(seed)

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