import numpy as np
import multiprocessing
import subprocess
import rebound as rb
from tqdm import tqdm
import itertools

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
# subprocess.run([compiler, './neomod3_simulator.f', '-o', neomod3_filename, '-O3'])

# generate debiased steady-state orbital distribution for both
size_min = 0.0079  # minimum size NEOMOD 3, km
size_max = 0.016  # maximum size NEOMOD 3, km
n_objs = 5000  # number of objects to generate
seed = int(np.random.default_rng().random() * -50202002)  # random seed

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

# constant bulk density
rho = 1500.

# initialize simulation
sim = rb.Simulation()
# sim.start_server(port=1236)  # start server
sim.integrator = 'trace'  # trace works well for these long-term simulations
# sim.ri_ias15.min_dt = 1e-8 # ensure that close encounters do not stall the integration 
sim.units = ('AU', 'kg', 'yr')  # units of distance, mass, time
# sim.dt = 1./20. * 1.0  # 1/50 of a year for whfast timestep
sim.t = 0.  # reset to 0
# Add Sun and Earth
sim.add(m = 1.988e30, r = 0.00465, hash = 'Sun')  # Sun
sim.add(m = 3.301e23, a=0.38709893, e=0.20564069, inc=0.122258, omega=0.50832, Omega=0.843587, r=1.63e-5, hash = 'Mercury')  # Mercury
sim.add(m=4.867e24, a=0.72333199, e=0.00677323, inc=0.05925, omega=0.957353, Omega=1.338331, r=4.05e-5, hash = 'Venus')  # Venus, angles in radians
sim.add(m = 5.97e24, a = 1.0, e = 0.01671123, inc = 9.e-7, omega = 1.9933, Omega = -0.1965352, r=4.258756e-5, hash = 'Earth')  # Earth, angles in radians
sim.add(m=6.417e23, a = 1.52366, e = 0.09341233, inc = 0.032299, omega = 4.99971, Omega = 0.865309, r=2.266e-5, hash = 'Mars')  # Mars
sim.add(m=1.898e27, a=5.20336301, e=0.04839266, inc=0.0227818, omega=-1.4975326, Omega=1.7550359, r=4.676e-4, hash = 'Jupiter')  # Jupiter, angles in radians
sim.N_active = sim.N  # 6 active particles, everything else is beyond the 10AU radius box
# define planets

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
            print(f'Asteroid mass: {j.m}, Asteroid diameter: {i.r}', file=f)
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


def mergeParticles(sim):
    # Find two closest particles
    min_d2 = 1e9 # large number
    ps = sim.particles
    sun, mercury, venus, earth, mars, jupiter = ps[0:6]
    for i1, i2 in itertools.combinations(range(sim.N), 2): # get all pairs of indices
        dp = ps[i1] - ps[i2]   # Calculates the coponentwise difference between particles 
        d2 = dp.x*dp.x+dp.y*dp.y+dp.z*dp.z
        if d2<min_d2:
            min_d2 = d2
            col_i1 = i1
            col_i2 = i2
    # check to see what collided
    if col_i1 == (0 or 2 or 3 or 6):
        with open(col_file, 'a') as f:
            print(f'{col_i2} collided with {col_i1} and was removed, seed:{seed}', file=f)
        # if Earth, add to arrays:
        if col_i1 == 3:
            # add to arrays
            earth = ps[col_i1]
            ast = ps[col_i2]
            earth_cart_positions.append(earth.xyz)
            earth_cart_vels.append(earth.vxyz)
            ast_cart_positions.append(ast.xyz)
            ast_cart_vels.append(ast.vxyz)
        # remove asteroid
        sim.remove(index=col_i2)
    elif col_i2 == (0 or 2 or 3 or 6):
        with open(col_file, 'a') as f:
            print(f'{col_i1} collided with {col_i2} and was removed, seed:{seed}', file=f)
        # if Earth, add to arrays:
        if col_i2 == 3:
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
        with open(col_file, 'a') as f:
            print(f'two asteroids collided? nothing removed, seed:{seed}', file=f)

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
sim.configure_box(20.)  # 10 AU on each side
sim.boundary = 'open'
# collisions
sim.collision = 'direct'  # trace must use direct, could change this with a different integrator
sim.collision_resolve = my_merge
sim.collision_resolve_keep_sorted = 1

# set up simulationarchive and have it save every 500 years
sim.save_to_file(f'./rebound_archives/rebound_archive_500_{seed}.bin', interval=500., delete_file=False)  # every hour

# collision file
col_file = 'collisions.txt'

# integrate, keep recording positions and velocities
times = np.linspace(sim.t, 1.e8, int(1e9))  # integrate

for t in tqdm(times):
    sim.integrate(t, exact_finish_time = 0)
# print the total number of particles remaining that haven't collided with the earth (should be quite small?)

# save
np.save('earth_cart_positions', np.array(earth_cart_positions))
np.save('earth_cart_vels', np.array(earth_cart_vels))
np.save('ast_cart_positions', np.array(ast_cart_positions))
np.save('ast_cart_vels', np.array(ast_cart_vels))