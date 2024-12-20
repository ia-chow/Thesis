import numpy as np
import pandas as pd
import scipy
import multiprocessing
import matplotlib.pyplot as plt
import os
import subprocess
import rebound as rb
from tqdm import tqdm
import itertools


# initialize arrays to record cartesian positions and velocities at impact
earth_cart_positions = []
earth_cart_vels = []
ast_cart_positions = []
ast_cart_vels = []

def my_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents # retreive the standard simulation object
    ps = sim.particles # easy access to list of particles
    i = ps[collided_particles_index.p1]   # Note that p1 < p2 is not guaranteed.    
    j = ps[collided_particles_index.p2]
    # record positions and velocities and remove
    print(i.hash, j.hash)
    # save this to a bin file so it can be loaded later
    # sim.save_to_file(f'sim_particles_{i.hash}{j.hash}_col_walltime.bin')
    if i.hash == earth.hash:  # if i is the earth
        print('Collision with Earth')
        earth = ps[i]
        ast = ps[j]
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(ast.xyz)
        ast_cart_vels.append(ast.vxyz)
        return 2  # remove particles with index j
    elif j.hash == earth.hash:  # if j is the earth
        print('Collision with Earth')
        earth = ps[j]
        ast = ps[i]
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(ast.xyz)
        ast_cart_vels.append(ast.vxyz)
        return 1  # remove particles with index i
    elif i.hash == sun.hash:
        print('Collision with the Sun')
        return 2  # remove particle with index j
    elif j.hash == sun.hash:
        print('Collision with the Sun')
        return 1
    elif i.hash == jupiter.hash:
        print('Collision with Jupiter')
        return 2   # remove index j
    elif j.hash == jupiter.hash:
        print('Collision with Jupiter')
        return 1  # remove index i
    elif i.hash == venus.hash:
        print('Collision with Venus')
        return 2
    elif j.hash == venus.hash:
        print('Collision with Venus')
        return 1
    else:  # continue simulation
        print('two asteroids collided? this shouldn\'t happen...')
        return 0  # continue simulation


def mergeParticles(sim):
    # Find two closest particles
    min_d2 = 1e9 # large number
    ps = sim.particles
    for i1, i2 in itertools.combinations(range(sim.N), 2): # get all pairs of indices
        dp = ps[i1] - ps[i2]   # Calculates the coponentwise difference between particles 
        d2 = dp.x*dp.x+dp.y*dp.y+dp.z*dp.z
        if d2<min_d2:
            min_d2 = d2
            col_i1 = i1
            col_i2 = i2
    # check to see what collided
    if col_i1.hash == (sun.hash or venus.hash or earth.hash or jupiter.hash):
        print(f'{col_i2.hash} collided with {col_i1.hash} and was removed')
        # if Earth, add to arrays:
        if col_i1.hash == earth.hash:
            # add to arrays
            earth = ps[col_i1]
            ast = ps[col_i2]
            earth_cart_positions.append(earth.xyz)
            earth_cart_vels.append(earth.vxyz)
            ast_cart_positions.append(ast.xyz)
            ast_cart_vels.append(ast.vxyz)
        # remove asteroid
        sim.remove(index=col_i2)
    elif col_i2.hash == (sun.hash or venus.hash or earth.hash or jupiter.hash):
        print(f'{col_i1.hash} collided with {col_i2.hash} and was removed')
        # if Earth, add to arrays:
        if col_i2.hash == earth.hash:
            # add to arrays
            earth = ps[col_i2]
            ast = ps[col_i1]
            earth_cart_positions.append(earth.xyz)
            earth_cart_vels.append(earth.vxyz)
            ast_cart_positions.append(ast.xyz)
            ast_cart_vels.append(ast.vxyz)
        # remove asteroid
        sim.remove(index=col_i1)
    else:
        print('two asteroids collided? nothing removed')

# initialize simulation
sim = rb.Simulation('rebound_archive_walltime.bin', snapshot=-1)  # start from the most recent snapshot
sim.start_server(port=1235)  # start server
sun, mercury, venus, earth, mars, jupiter = sim.particles[0:6]
sim.collision = 'direct'
sim.collision_resolve = my_merge
sim.collision_resolve_keep_sorted = 1

# set up simulationarchive and have it save every two hours of real time
sim.save_to_file('rebound_archive_walltime.bin', walltime=3600, delete_file=False)  # every hour

# integrate, keep recording positions and velocities
times = np.linspace(sim.t, 1.e8, int(1e9))  # integrate

for t in tqdm(times):
    try:
        sim.integrate(t, exact_finish_time = 0)
    except rb.Encounter:
        mergeParticles(sim)
# print the total number of particles remaining that haven't collided with the earth (should be quite small?)

# save
np.save('earth_cart_positions', np.array(earth_cart_positions))
np.save('earth_cart_vels', np.array(earth_cart_vels))
np.save('ast_cart_positions', np.array(ast_cart_positions))
np.save('ast_cart_vels', np.array(ast_cart_vels))