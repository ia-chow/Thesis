import numpy as np
import pandas as pd
import scipy
import multiprocessing
import matplotlib.pyplot as plt
import os
import subprocess
import rebound as rb
from tqdm import tqdm

# constants for earth and sun
a0 = 1.  # semi-major axis of earth in au
m_earth = 5.97e24  # mass of earth in kg
R_earth = 4.259e-5  # radius of earth in au
M_sun = 1.988e30  # mass of sun in kg
e0 = 0.01671123  # eccentricitiy of earth

compiler = 'gfortran'  # name of fortran compiler to use
# compile fortran code if it hasn't been already
subprocess.run([compiler, './Pokorny_et_al_2013/PROB_KOZAI_V001.f90', '-o', './Pokorny_et_al_2013/CODE', '-O3'])

# OPIK PROBABILITY PER REVOLUTION
def get_opik_annual_P_packed(aei, a0=a0, m=m_earth, R=R_earth, M=M_sun):
    """
    Get the collision probability per revolution P for an object with semi-major axis a in AU, eccentricity e, and inclination i in degrees
    and a target (Earth) on a circular orbit with constant semi-major axis a0 in AU, mass m in kg, and radius R in AU, 
    both orbiting a star with mass M in kg

    return: P, the collision probability per revolution
    """
    a, e, i = aei
    Q = R/a0  # Q
    Ux = np.sqrt(2 - (1/a) - (a * (1 - (e ** 2))))  # Ux
    U = np.sqrt(3 - (1/a) - ((2 * np.sqrt(a * (1 - e ** 2))) * np.cos(np.deg2rad(i))))  # U
    tau = Q * np.sqrt(1 + ((2 * m)/(M * Q * (U ** 2))))  # tau
    # tau = R * np.sqrt(1 + R/Q)
    # compute P
    P = ((tau ** 2) * U)/(np.pi * np.sin(np.deg2rad(i)) * np.abs(Ux))
    # return
    return P


# define function
def get_pokorny_annual_P_packed(aeiperi, a_earth=a0, e_earth=e0, R_earth=R_earth):
    """
    Get the collisional probability using Petr Pokorny's 2013 code with a, e, i, peri packed as a tuple
    Takes in a given a, e, i and argument of pericenter of the projectile and returns the annual collisional probability

    a is in units of AU, i is in units of degrees, argument of pericenter is in degrees
    """
    a, e, i, peri = aeiperi  # unpack
    # run fortran code
    output = subprocess.run(['./Pokorny_et_al_2013/CODE', '<'], 
                            input=f'{a},{e},{i},{peri}\n{a_earth}\n{e_earth}'.encode('utf-8'), 
                            capture_output=True)
    # print(output)
    col_prob = np.float64(output.stdout.decode('utf-8').split()[-1]) * (R_earth ** 2)
    return col_prob

# compile the NEOMOD3 model
neomod3_filename = './NEOMOD3_CODE'
# compile neomod3 fortran code
subprocess.run([compiler, './neomod3_simulator.f', '-o', neomod3_filename, '-O3'])

# generate debiased steady-state orbital distribution for both
size_min = 0.0079  # minimum size NEOMOD 3, km
size_max = 0.016  # maximum size NEOMOD 3, km
n_objs = 10000  # number of objects to generate
seed = -50202002  # seed to use

# generate neomod3 output:
# Returns H, a (AU), e, i (degrees), diameter (km), albedo
neomod3_output = np.array(subprocess.run([neomod3_filename, '<'], 
                                input=f'input_neomod3.dat\n{seed}\n{n_objs}\n{size_min} {size_max}'.encode('utf-8'), 
                                capture_output=True).stdout.decode('utf-8').split(), dtype=np.float64).reshape(n_objs, 6)  # 6 parameters

# unpack values
neomod3_hs, neomod3_as, neomod3_es, neomod3_is, neomod3_sizes, neomod3_albs = neomod3_output.T

def peri_intersect(a, e, r_t=a0):
    """
    Computes the two possible arguments of pericenter required for an object with a given a, e to 
    intersect the orbit of a planet on a circular orbit with semi-major axis r_t and randomly picks one.
    Returns argument of pericenter in degrees
    """
    # compute base case
    base = np.arccos((a * (1. - e ** 2) - r_t)/(e * r_t))
    # other case is offset by 180 degrees, randomly pick one of them and convert to degrees
    return np.rad2deg(np.c_[0. - base, np.pi - base][np.arange(len(base)), np.random.randint(0, 2, size=len(base))])

neomod3_peris = peri_intersect(neomod3_as, neomod3_es)

# compute impact probabilities:
neomod3_aeis = np.c_[neomod3_as, neomod3_es, neomod3_is]
neomod3_aeiperis = np.c_[neomod3_as, neomod3_es, neomod3_is, neomod3_peris]
# multiprocess, computing for neomod2 and 3
pool = multiprocessing.Pool()
# neomod3_pts_pokorny = np.array(list(tqdm(pool.imap(get_pokorny_annual_P_packed, neomod3_aeiperis), total = len(neomod3_aeiperis))))
neomod3_pts = np.array(list(tqdm(pool.imap(get_opik_annual_P_packed, neomod3_aeis), total = len(neomod3_aeis))))
# join and close
pool.close()
pool.join()

def filter_obj_mask(pts, qs, Qs, ip_threshold=5.e-8, target_a = 1.):
    """
    Return mask to truncate objects at ip threshold ip_threshold, remove 0 and nan values, 
    and remove objects that do not cross trarget's orbit
    """
    # mask
    return (0. < pts) & (pts < ip_threshold) & ~np.isnan(pts) & (qs < target_a) & (Qs > target_a)


neomod3_qs = neomod3_as * (1. - neomod3_es)
neomod3_Qs = neomod3_as * (1. + neomod3_es)
# mask for 0, nan, impact probability, q and Q
mask = filter_obj_mask(neomod3_pts, neomod3_qs, neomod3_Qs)
# apply to pts and qs
m_neomod3_pts = neomod3_pts[mask]
m_neomod3_qs = neomod3_qs[mask]

m_neomod3_output = neomod3_output[mask]  # filter neomod3 output
m_neomod3_hs, m_neomod3_as, m_neomod3_es, m_neomod3_is, m_neomod3_sizes, m_neomod3_albs = m_neomod3_output.T
m_neomod3_peris = neomod3_peris[mask]
# Randomly generate ascending node between 0 and 360 degrees?
m_neomod3_Omegas = np.random.uniform(0., 360., len(m_neomod3_output)) 
# assemble required parameters into an array
m_neomod3_params = np.c_[m_neomod3_as, m_neomod3_es, m_neomod3_is, m_neomod3_peris, m_neomod3_Omegas, m_neomod3_sizes * 1000.]  # convert sizes to m

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
    sim.save_to_file(f'sim_particles_{i.hash}{j.hash}_col.bin')
    if i.hash == 'Earth':  # if i is the earth
        print('Collision with Earth')
        earth = ps[i]
        ast = ps[j]
        earth_cart_positions.append(earth.xyz)
        earth_cart_vels.append(earth.vxyz)
        ast_cart_positions.append(ast.xyz)
        ast_cart_vels.append(ast.vxyz)
        return 2  # remove particles with index j
    elif j.hash == 'Earth':  # if j is the earth
        print('Collision with Earth')
        earth = ps[j]
        ast = ps[i]
        earth_cart_positions[i] = earth.xyz
        earth_cart_vels[i] = earth.vxyz
        ast_cart_positions[i] = ast.xyz
        ast_cart_vels[i] = ast.vxyz
        return 1  # remove particles with index i
    elif i.hash == 'Sun':
        print('Collision with the Sun')
        return 2  # remove particle with index j
    elif j.hash == 'Sun':
        print('Collision with the Sun')
        return 1
    elif i.hash == 'Venus':
        print('Collision with Venus')
        return 2
    elif j.hash == 'Venus':
        print('Collision with Venus')
        return 1
    else:  # if neither is the earth, continue simulation
        print('two asteroids collided? this shouldn\'t happen...')
        return 0  # continue simulation

# constant bulk density
rho = 1500.

# initialize simulation
sim = rb.Simulation()
sim.integrator = 'trace'  # trace, symplectic
sim.units = ('AU', 'kg', 'yr')  # units of distance, mass, time
sim.collision = 'line'
sim.collision_resolve = my_merge
sim.collision_resolve_keep_sorted = 1
# sim.dt = 1./20. * 1.0  # 1/20 of a year for whfast timestep
sim.t = 0.  # reset to 0
# Add Sun and Earth
sim.add(m = 1.988e30, r = 0.00465, hash = 'Sun')  # Sun
sim.add('Mercury', hash = 'Mercury')
sim.add('Venus', hash = 'Venus')
sim.add(m = 5.97e24, a = 1.0, e = 0.01671123, inc = 9.e-7, omega = 1.9933, Omega = -0.1965352, r=4.258756e-5, hash = 'Earth')  # Earth, angles in radians
sim.add('Mars', hash = 'Mars')
sim.add('Jupiter', hash = 'Jupiter')
sim.add('Saturn', hash = 'Saturn')
sim.add('Uranus', hash = 'Uranus')
sim.add('Neptune', hash = 'Neptune')
sim.N_active = sim.N  # 8 active particles
# define planets
sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune = sim.particles
# asteroids are test particles
sim.testparticle_type = 1
# add all the asteroids as non active particles so they don't take up so much processing power
for j, params in enumerate(m_neomod3_params):
    a_a, e_a, i_a_deg, omega_a_deg, Omega_a_deg, size_a = params
    i_a, omega_a, Omega_a = map(np.radians, [i_a_deg, omega_a_deg, Omega_a_deg])  # convert to radians
    m_a = 4. * rho * (size_a/2.) ** 3  # assume bulk density of rho, size is in meters so this is units of kg
    sim.add(m = m_a, a = a_a, e = e_a, omega = omega_a, inc = i_a, Omega = Omega_a, r=(size_a * 6.685e-12), hash = f'particle_{j}') 
# move to com
sim.move_to_com()
# set maximum distance for particle to be removed
sim.configure_box(10.)  # 10 AU
sim.boundary = 'open'

# set up simulationarchive and have it save every 100000 years
sim.save_to_file('rebound_archive.bin', interval=1e5, delete_file=True)

# integrate until there is a collision and keep recording positions and velocities
sim.integrate(1.e8, exact_finish_time=0)  # 5e10 years, 0 for symplectic integrator
# print the total number of particles remaining that haven't collided with the earth (should be quite small?)


# save
np.save('earth_cart_positions', earth_cart_positions)
np.save('earth_cart_vels', earth_cart_vels)
np.save('ast_cart_positions', ast_cart_positions)
np.save('ast_cart_vels', ast_cart_vels)