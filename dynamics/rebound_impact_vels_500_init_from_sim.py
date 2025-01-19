import numpy as np
import rebound as rb


# initialize arrays to record cartesian positions and velocities at impact
earth_cart_positions = []
earth_cart_vels = []
ast_cart_positions = []
ast_cart_vels = []

# constant bulk density
rho = 1500.

def my_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents # retreive the standard simulation object
    ps = sim.particles # easy access to list of particles
    sun, mercury, venus, earth, mars, jupiter = ps[0:6]
    i = ps[collided_particles_index.p1]   # Note that p1 < p2 is not guaranteed.    
    j = ps[collided_particles_index.p2]
    # record positions and velocities and remove
    with open(col_file, 'a') as f:
        print(f'particle index:{i.index},{j.index}', file=f)
        print(f'Simulation time:{sim.t}, Simulation number of particles:{sim.N}', file=f)
    # save this to a bin file so it can be loaded later
    # sim.save_to_file(f'sim_particles_{i.hash}{j.hash}_col_walltime.bin')
    if i == earth:  # if i is the earth
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(j.xyz)
        ast_cart_vels.append(j.vxyz)
        with open(col_file, 'a') as f:
            print(f'Collision with Earth, seed:{seed}', file=f)
            print(f'Asteroid mass: {j.m}, Asteroid diameter: {j.r}', file=f)
            print(f'Asteroid orbital elements: a {j.a}, e {j.e}, i {j.inc}, omega {j.omega}, Omega {j.Omega}, M {j.M}', file=f)
            print(f'Earth position {earth.xyz}, Earth velocity {earth.vxyz}, Asteroid position {j.xyz}, Asteroid velocity {j.vxyz}', file=f)
        return 2  # remove particles with index j
    elif j == earth:  # if j is the earth
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(i.xyz)
        ast_cart_vels.append(i.vxyz)
        with open(col_file, 'a') as f:
            print(f'Collision with Earth, seed:{seed}', file=f)
            print(f'Asteroid mass: {i.m}, Asteroid diameter: {i.r}', file=f)
            print(f'Asteroid orbital elements: a {i.a}, e {i.e}, i {i.inc}, omega {i.omega}, Omega {i.Omega}, M {i.M}', file=f)
            print(f'Earth position {earth.xyz}, Earth velocity {earth.vxyz}, Asteroid position {i.xyz}, Asteroid velocity {i.vxyz}', file=f)
        return 1  # remove particles with index i
    elif i == sun:
        with open(col_file, 'a') as f:
            print(f'Collision with the Sun, seed:{seed}', file=f)
            # print(f'Sun position {sun.xyz}, Sun velocity {sun.vxyz}, Asteroid position {ast.xyz}, Asteroid velocity {ast.vxyz}')
        return 2  # remove particle with index j
    elif j == sun:
        with open(col_file, 'a') as f:
            print(f'Collision with the Sun, seed:{seed}', file=f)
            # print(f'Sun position {sun.xyz}, Sun velocity {sun.vxyz}, Asteroid position {ast.xyz}, Asteroid velocity {ast.vxyz}')
        return 1  # remove index i
    elif i == jupiter:
        with open(col_file, 'a') as f:
            print(f'Collision with Jupiter, seed:{seed}', file=f)
        return 2   # remove index j
    elif j == jupiter:
        with open(col_file, 'a') as f:
            print(f'Collision with Jupiter, seed:{seed}', file=f)
        return 1  # remove index i
    elif i == venus:
        with open(col_file, 'a') as f:
            print(f'Collision with Venus, seed:{seed}', file=f)
        return 2
    elif j == venus:
        with open(col_file, 'a') as f:
            print(f'Collision with Venus, seed:{seed}', file=f)
        return 1
    elif i == mercury:
        with open(col_file, 'a') as f:
            print(f'Collision with Mercury, seed:{seed}', file=f)
        return 2
    elif j == mercury:
        with open(col_file, 'a') as f:
            print(f'Collision with Mercury, seed:{seed}', file=f)
        return 1
    elif i == mars:
        with open(col_file, 'a') as f:
            print(f'Collision with Mars, seed:{seed}', file=f)
        return 2
    elif j == mars:
        with open(col_file, 'a') as f:
            print(f'Collision with Mars, seed:{seed}', file=f)
        return 1
    else:  # continue simulation
        with open(col_file, 'a') as f:
            print(f'two asteroids collided?, seed:{seed}', file=f)
        return 0  # continue simulation


#### CHANGE THIS TO THE SEED NUMBER SIMULATION BEING INITIALIZED

# NEED TO FIND SOME NODE TO RUN THESE ON:

# seed = -33894958

def my_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents # retreive the standard simulation object
    ps = sim.particles # easy access to list of particles
    sun, mercury, venus, earth, mars, jupiter = ps[0:6]
    i = ps[collided_particles_index.p1]   # Note that p1 < p2 is not guaranteed.    
    j = ps[collided_particles_index.p2]
    # record positions and velocities and remove
    with open(col_file, 'a') as f:
        print(f'particle index:{i.index},{j.index}, seed:{seed}', file=f)
        print(f'Simulation time:{sim.t}, Simulation number of particles:{sim.N}', file=f)
    # save this to a bin file so it can be loaded later
    # sim.save_to_file(f'sim_particles_{i.hash}{j.hash}_col_walltime.bin')
    if i == earth:  # if i is the earth
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(j.xyz)
        ast_cart_vels.append(j.vxyz)
        with open(col_file, 'a') as f:
            print(f'Collision with Earth, seed:{seed}', file=f)
            print(f'Asteroid mass: {j.m}, Asteroid diameter: {j.r}', file=f)
            print(f'Asteroid orbital elements: a {j.a}, e {j.e}, i {j.inc}, omega {j.omega}, Omega {j.Omega}, M {j.M}', file=f)
            print(f'Earth position {earth.xyz}, Earth velocity {earth.vxyz}, Asteroid position {j.xyz}, Asteroid velocity {j.vxyz}', file=f)
        return 2  # remove particles with index j
    elif j == earth:  # if j is the earth
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(i.xyz)
        ast_cart_vels.append(i.vxyz)
        with open(col_file, 'a') as f:
            print(f'Collision with Earth, seed:{seed}', file=f)
            print(f'Asteroid mass: {i.m}, Asteroid diameter: {i.r}', file=f)
            print(f'Asteroid orbital elements: a {i.a}, e {i.e}, i {i.inc}, omega {i.omega}, Omega {i.Omega}, M {i.M}', file=f)
            print(f'Earth position {earth.xyz}, Earth velocity {earth.vxyz}, Asteroid position {i.xyz}, Asteroid velocity {i.vxyz}', file=f)
        return 1  # remove particles with index i
    elif i == sun:
        with open(col_file, 'a') as f:
            print(f'Collision with the Sun, seed:{seed}', file=f)
            # print(f'Sun position {sun.xyz}, Sun velocity {sun.vxyz}, Asteroid position {ast.xyz}, Asteroid velocity {ast.vxyz}')
        return 2  # remove particle with index j
    elif j == sun:
        with open(col_file, 'a') as f:
            print(f'Collision with the Sun, seed:{seed}', file=f)
            # print(f'Sun position {sun.xyz}, Sun velocity {sun.vxyz}, Asteroid position {ast.xyz}, Asteroid velocity {ast.vxyz}')
        return 1  # remove index i
    elif i == jupiter:
        with open(col_file, 'a') as f:
            print(f'Collision with Jupiter, seed:{seed}', file=f)
        return 2   # remove index j
    elif j == jupiter:
        with open(col_file, 'a') as f:
            print(f'Collision with Jupiter, seed:{seed}', file=f)
        return 1  # remove index i
    elif i == venus:
        with open(col_file, 'a') as f:
            print(f'Collision with Venus, seed:{seed}', file=f)
        return 2
    elif j == venus:
        with open(col_file, 'a') as f:
            print(f'Collision with Venus, seed:{seed}', file=f)
        return 1
    elif i == mercury:
        with open(col_file, 'a') as f:
            print(f'Collision with Mercury, seed:{seed}', file=f)
        return 2
    elif j == mercury:
        with open(col_file, 'a') as f:
            print(f'Collision with Mercury, seed:{seed}', file=f)
        return 1
    elif i == mars:
        with open(col_file, 'a') as f:
            print(f'Collision with Mars, seed:{seed}', file=f)
        return 2
    elif j == mars:
        with open(col_file, 'a') as f:
            print(f'Collision with Mars, seed:{seed}', file=f)
        return 1
    else:  # continue simulation
        with open(col_file, 'a') as f:
            print(f'two asteroids collided?, seed:{seed}', file=f)
        return 0  # continue simulation


# initialize simulation from archive
sim = rb.Simulation(f'./rebound_archives/rebound_archive_500_{seed}.bin')  # start from the most recent snapshot
# sim.start_server(port=1236)  # start server

# set maximum distance for particle to be removed
sim.configure_box(20.)  # 10 AU on each side
sim.boundary = 'open'
# collisions THIS HAS TO BE HERE OTHERWISE IT DOESN'T WORK
sim.collision = 'direct'  # trace must use direct, could change this with a different integrator
sim.collision_resolve = my_merge
sim.collision_resolve_keep_sorted = 1

# collision file
col_file = 'collisions.txt'

# set up simulationarchive and have it save every 500 years
sim.save_to_file(f'./rebound_archives/rebound_archive_500_{seed}.bin', interval=500., delete_file=False)  # every hour

# integrate, keep recording positions and velocities
sim.integrate(1.e8, exact_finish_time = 0)

# print the total number of particles remaining that haven't collided with the earth (should be quite small?)

# save
np.save('earth_cart_positions', np.array(earth_cart_positions))
np.save('earth_cart_vels', np.array(earth_cart_vels))
np.save('ast_cart_positions', np.array(ast_cart_positions))
np.save('ast_cart_vels', np.array(ast_cart_vels))