import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import subprocess
from sklearn.neighbors import KernelDensity  # kernel density estimation

# NEED TO HAVE THE FILES 'cneos_fireball_data.csv' and 'usg-ground-based-comparison/usg-ground-based-comparison_EDITED.csv' in the required directories for this script

# read in cneos fireballs, dropping row if any of time, latitude, longitude, altitude, vx, vy or vz are nan
cneos_fireballs_raw = pd.read_csv('cneos_fireball_data.csv').dropna(subset=['Peak Brightness Date/Time (UT)', 
                                                                            'Latitude (deg.)', 'Longitude (deg.)', 
                                                                            'Altitude (km)', 'vx', 'vy', 'vz'])
# MANUALLY REMOVE THE DECAMETER-SIZED IMPACTORS LATER
# remove decameter-sized impactors (anything above 10 kt)
# cneos_fireballs_raw = cneos_fireballs_raw[cneos_fireballs_raw['Calculated Total Impact Energy (kt)'] <= 10.]
# 304 fireballs in total

# rtd is constant
rtd = np.pi/180  # rad/deg
# get time, converting to the format used by pylig: '%Y%m%d-%H%M%S.%f'
cneos_t = pd.to_datetime(cneos_fireballs_raw['Peak Brightness Date/Time (UT)'], format='mixed').dt.strftime('%Y%m%d-%H%M%S.%f')
# get elevation
cneos_elevation = cneos_fireballs_raw['Altitude (km)']

# get x, y, z velocities, dropping nans
cneos_vx, cneos_vy, cneos_vz = cneos_fireballs_raw.vx, cneos_fireballs_raw.vy, cneos_fireballs_raw.vz
# convert to numerical long, lat, dropping nans
cneos_latitude = (cneos_fireballs_raw['Latitude (deg.)'].str[:-1].astype(float) * (2 * (cneos_fireballs_raw['Latitude (deg.)'].str[-1:] == 'N') - 1))
cneos_longitude = (cneos_fireballs_raw['Longitude (deg.)'].str[:-1].astype(float) * (2 * (cneos_fireballs_raw['Longitude (deg.)'].str[-1:] == 'E') - 1))

# cneos_vx, cneos_vy, cneos_vz = -10.8, 1.2, -12.8
# cneos_latitude = 59.8
# cneos_longitude = 16.8

# should get 242.771 azim, 16.6202 zen, 16.7905 speed

# get vn, ve, vd from vx, vy, vz and lat, long
cneos_vn = -cneos_vx * np.sin(cneos_latitude * rtd) * np.cos(cneos_longitude * rtd) - cneos_vy * np.sin(cneos_longitude * rtd) * np.sin(cneos_latitude * rtd) + cneos_vz * np.cos(cneos_latitude * rtd)
cneos_ve = -cneos_vx * np.sin(cneos_longitude * rtd) + cneos_vy * np.cos(cneos_longitude * rtd)
cneos_vd = -cneos_vx * np.cos(cneos_latitude * rtd) * np.cos(cneos_longitude * rtd) - cneos_vy * np.cos(cneos_latitude * rtd) * np.sin(cneos_longitude * rtd) - cneos_vz * np.sin(cneos_latitude * rtd)
# get azimuth and zenith
cneos_azim = np.arctan2(cneos_ve, cneos_vn)/rtd + 180.  # NOTE that arctan2 in excel is (x, y) but arctan2 in numpy is (y, x)!
cneos_zen = np.arctan(np.sqrt(cneos_vn ** 2 + cneos_ve ** 2)/cneos_vd)/rtd
# get total velocity
cneos_v = np.sqrt(cneos_vx ** 2 + cneos_vy ** 2 + cneos_vz ** 2)

# filter
zen_mask = (0. < cneos_zen) & (cneos_zen < 90.)  # create mask for zenith between 0 and 90 degrees

# filter v, t, lat, lon, elevation, azim, zenith by the mask
cneos_v = cneos_v[zen_mask]
cneos_t = cneos_t[zen_mask]
cneos_latitude = cneos_latitude[zen_mask]
cneos_longitude = cneos_longitude[zen_mask]
cneos_elevation = cneos_elevation[zen_mask]
cneos_azim = cneos_azim[zen_mask]
cneos_zen = cneos_zen[zen_mask]

# skip the first two rows after header and the last three rows since those are garbage
# note that we want the usg-ground-based-comparison file to be in this form since we are removing absolute rows
usg_ground_based_comparison = pd.read_csv('usg-ground-based-comparison/usg-ground-based-comparison_EDITED.csv', 
                                          sep = ',', skip_blank_lines=True).iloc[2:-3]  

# forward fill Event and radiant, backward fill dv based on the xlsx file
usg_ground_based_comparison['Event'] = usg_ground_based_comparison['Event'].fillna(method='ffill')
usg_ground_based_comparison['Radiant Diff'] = usg_ground_based_comparison['Radiant Diff'].fillna(method='ffill')
usg_ground_based_comparison['DV'] = usg_ground_based_comparison['DV'].fillna(method='bfill')

# drop every other row since we don't need duplicate rows for event, radiant diff, or dv
# however we want to keep the usg reported speed, zenith angle and other parameters (second row for each event), 
# so we start by dropping the first row rather than the second
usg_ground_based_comparison = usg_ground_based_comparison.iloc[1::2]
# strip whitespace from headers to avoid errors with indexing later
usg_ground_based_comparison.columns = usg_ground_based_comparison.columns.str.strip()
# sort dataframe alphabetically inplace
usg_ground_based_comparison.sort_values('Event', inplace=True)

# convert Speed, DV, Radiant Diff, Radiant Zenith Angle, Height, Begin Height, End Height, Length to numeric to avoid errors
cols_to_convert = ['Speed', 'DV', 'Radiant Diff', 'Radiant Zenith Angle', 'Height', 'Begin Height (km)', 'End Height (km)', 'Length (km)']
usg_ground_based_comparison[cols_to_convert] = usg_ground_based_comparison[cols_to_convert].apply(pd.to_numeric, errors='coerce', axis=1)

# convert date to datetime object
usg_ground_based_comparison['Date'] = pd.to_datetime(usg_ground_based_comparison['Date'])

# usg speed, dv, drad
usg_speed = usg_ground_based_comparison['Speed']
# speed and radiant uncertainties
dv = usg_ground_based_comparison['DV']
drad = usg_ground_based_comparison['Radiant Diff']

#### KDES

# Speed

# parameters for the KDE fitting
dv_ub = 8.  # upper bound for fitting kde
dv_lb = -8.  # lower bound for fitting kde
dv_bandwidth = 0.7  # this is finicky  # 0.8 used by default
kernel_type = 'gaussian'  # use gaussian kernel

# setting up the values
values_dv = np.linspace(dv_lb, dv_ub, int(1e4))
kde_dv = np.array(dv).reshape((len(dv), 1))
kde_values_dv = values_dv.reshape(len(values_dv), 1)

# doing the KDE fit
kde_model_dv = KernelDensity(bandwidth=dv_bandwidth, kernel=kernel_type)
kde_model_dv.fit(kde_dv)

# parameters for the KDE fitting
drad_lb = 0.  # upper bound for fitting kde
drad_ub = 2.  # lower bound for fitting kde
log_drad_bandwidth = 0.15  # this is finicky  # 0.15 used by default

# Radiant

# setting up the values
log_drad = np.log10(drad)  # convert to log
values_drad = np.linspace(drad_lb, drad_ub, int(1e4))
kde_log_drad = np.array(log_drad).reshape((len(log_drad), 1))
kde_values_log_drad = values_drad.reshape(len(values_drad), 1)

# kde fit
kde_model_log_drad = KernelDensity(bandwidth=log_drad_bandwidth, kernel=kernel_type)  # use the same kernel as for the dv kde
kde_model_log_drad.fit(kde_log_drad)

# list of orbital parameters we want to get (named the same as in the wmpl output, in that order
orb_param_variables = ['a', 'e', 'i', 'peri', 'node', 'M']  
# number of Monte Carlo samples for each event
n_monte_carlo_samples = 1000  # use 1000 MC samples

### HELPER FUNCTION to get the zenith/azimuth from the radiant uncertainty
def get_zen_azim_diff(zen, azim, theta, n_samples):
    """
    Get n_samples random new radiants that are a fixed angle theta away from the radiant given by (zen, azim) in 3D
    Note that zenith angle is 90 degrees - azimuth angle

    param zen: zenith angle, in degrees
    param azim: azimuth angle, in degrees
    param theta: angle from radiant that we want our new radiants to be, in degrees
    param n_samples: number of new radiants we want

    return zen_diff, azim_diff: tuple of arrays of floats of size n_samples each: 
    zen_diff and azim_diff are the new randomly generated radiant zenith/azimuth minus the given radiant zenith/azimuth
    """
    new_zens_deg = np.array([np.nan])  # initialize this to get the while loop started

    # this is a while loop to reject any zenith distance above 90 or below 0:
    while not np.all((new_zens_deg > 0.) & (new_zens_deg < 90.)):
        phis = np.random.uniform(0., 2 * np.pi, size=n_samples)  # these are the random directions we want our radiants to be in

        # convert zenith and azimuth into cartesian coordinates x, y, z
        v1 = np.array([np.sin(np.deg2rad(zen)) * np.cos(np.deg2rad(azim)), 
                       np.sin(np.deg2rad(zen)) * np.sin(np.deg2rad(azim)), 
                       np.cos(np.deg2rad(zen))])
        # get random new normalized vectors that are an angle theta away from v1, following the procedure described here:
        # https://math.stackexchange.com/questions/2057315/parameterising-all-vectors-a-fixed-angle-from-another-vector-in-3d
        # standard basis vector to use for generating orthonormal basis
        standard_basis_vec = np.array([1, 0, 0]) 
        # generate two basis vectors orthonormal to v1
        # normalize v1 in case it isn't already
        v1_normed = v1/np.linalg.norm(v1)
        # generate v2
        v2 = np.cross(v1_normed, standard_basis_vec)
        # normalize it
        v2_normed = v2/np.linalg.norm(v2)
        # generate a normalized v3 from the normalized v1 and v2
        v3_normed = np.cross(v1_normed, v2_normed)
        # now generate new vectors with radiant difference scattered by phis using vectorized operations
        new_radiants = v1_normed + np.tan(np.deg2rad(theta)) * (np.multiply.outer(np.cos(phis), v2_normed) + 
                                                                np.multiply.outer(np.sin(phis), v3_normed))
        # normalize the new vectors
        new_radiants_normed = new_radiants/np.linalg.norm(new_radiants, axis=1).reshape(-1, 1)
        # convert from cartesian coordinates back to zenith/azimuth coordinates
        xs, ys, zs = new_radiants_normed.T
        # get zeniths from cartesian
        new_zens = np.arccos(zs)
        # get azimuths from cartesian
        new_azims = np.arcsin(ys/np.sin(new_zens))
        # convert to degrees
        new_zens_deg = np.rad2deg(new_zens)
        new_azims_deg = np.rad2deg(new_azims)
        # first, manually convert the azimuths to the right quadrant since they can range from [0, 2pi] while zeniths can only be [0, pi/2]
        # Q1
        if 0. <= azim <= 90.:
            new_azims_deg = new_azims_deg  # already correct
        # Q2 and Q3:
        if 90. < azim <= 270.:
            new_azims_deg = np.abs(new_azims_deg - 180.)  # convert 
        # Q4:
        if 270. < azim <= 360.:
            new_azims_deg += 360  # convert
        
        # # finally, convert any negative azimuths back to [0, 360] degrees by adding 360 to any values less than 0
        # new_azims_deg += 360 * (new_azims_deg < 0)
        # # and convert any azimuths above 360 back down to [0, 360] as well
        # new_azims_deg = np.mod(new_azims_deg, 360)
            
    # return zenith and azimuth difference
    return new_zens_deg - zen, new_azims_deg - azim


# draw a monte carlo sample and get the orbital parameters using WMPL
def get_monte_carlo_orbital_parameters(kdes_state_vector_params, orb_param_variables=orb_param_variables, r_sun=0.00465):
    """
    Samples a Monte Carlo sample from provided KDEs for speed and radiant difference, and gets 
    the orbital parameter values corresponding to each parameter in orb_param_variables from the state vector state_vector
    
    param kdes_state_vector_params: tuple of (dv_kde, drad_kde, v, t, a, o, e, azim, alt):
    dv_kde is the speed uncertainty kde: should be a scikit-learn KernelDensity object
    log_drad_kde is the log10-radiant uncertainty kde: should be a scikit-learn KernelDensity object
    v is the velocity in km/s
    t is the time of the event in the form that the wmpl.Trajectory.Orbit function takes in, '%Y%m%d-%H%M%S.%f'
    a is the latitude, in degrees N
    o is the longitude, in degrees E
    e is the elevation of the event, in km
    azim is the radiant azimuth, in degrees
    zen is the zenith distance (90 minus the altitude angle), in degrees

    returns a 1-D array of parameter values corresponding to the parameters in orb_param_variables, in the same order
    """
    orb_param_array = np.array([])  # initialize orb_param_array as empty array so it doesn't have a size to start
    ecc = np.nan  # initialize both of these as nans so they don't have a size to start
    sma = np.nan

    # unpack kdes and state_vector_params
    dv_kde, log_drad_kde, v, t, a, o, e, azim, zen = kdes_state_vector_params  # get the kdes and state vector parameters
    # converting from zenith angle to altitude angle
    alt = 90 - zen
    
    #### THIS IS A WHILE LOOP AS A SAFEGUARD BECAUSE IT SEEMS LIKE SOMETIMES WMPL DOESN'T RETURN THE ORBITAL ELEMENTS AT ALL

    # if orbital parameter array is empty (no orbital elements returned), OR eccentricity is greater than 0.9, OR q is inside the radius of the sun, 
    # then we repeat this until none of those 3 are true
    while (not orb_param_array.size or ecc > 0.98 or sma * (1. - ecc) < r_sun):
        np.random.seed(None)  # explicitly reset the seed since the sampler for sklearn doesn't seem to be sampling randomly otherwise
        # get monte carlo sample of the speed and log-radiant uncertainty
        dv = dv_kde.sample(n_samples=1).flatten()  # get a single dv mc sample
        log_drad = log_drad_kde.sample(n_samples=1).flatten()  # get a single log drad mc sample

        # convert the log-radiant uncertainty into zenith and azimuth uncertainty using the helper function
        # with the restrictions that zenith must be between [0, 90] degrees
        dzen, dazim = get_zen_azim_diff(zen, azim, 10 ** log_drad, n_samples=1)  # log10-radiant so convert back to radiant
        # difference in altitude is negative of difference in zenith since it's 90 - zenith
        # so zen1 - zen2 = (90 - alt1) - (90 - alt2) = alt2 - alt1
        dalt = -dzen
        # adding the uncertainty in velocity to the velocity vector of the event
        # and adding the uncertainty in zenith angle and azimuth angle to the radiant
        # convert the singleton arrays to floats as well
        new_v = v + dv[0]
        new_alt = alt + dalt[0]
        # azimuth is modulo 360 to keep it between 0 and 360 degrees
        new_azim = (azim + dazim[0]) % 360
        # get the output from the wmpl.Trajectory.Orbit script
        output = subprocess.run(['python3', '-m', 'wmpl.Trajectory.Orbit', '-v', f'{new_v}', '-t', f'{t}', '-a', f'{a}', 
                                 '-o', f'{o}', '-e', f'{e}', '--azim', f'{new_azim}', '--alt', f'{new_alt}', 
                                 '--vrotcorr', '--statfixed'], capture_output=True)
        # get the orbital parameters from the output in the order of orb_param_variables
        
        # orb_param_list = [elem for elem in list(map(str.strip, output.stdout.decode('utf-8').split('\n'))) 
        #               if elem.startswith(tuple(orb_param_variables))]
        # orb_param_array = [np.float64(string) for param in orb_param_list 
        #                    for string in param.split() if string.lstrip('-').replace('.', '').isdigit() or string =='nan']]
        
        orb_param_array = np.array([np.float64(string) for param in [elem for elem in 
                                                                     list(map(str.strip, output.stdout.decode('utf-8').split('\n'))) 
                                                                     if elem.startswith(tuple(orb_param_variables))] 
                                    for string in param.split() if string.lstrip('-').replace('.', '').isdigit() or string =='nan'])

        # if it's nonempty then unpack orbital parameter array to update semi major axis and eccentricity for the while loop
        if orb_param_array.size:
            sma, ecc, inc, peri, node, M = orb_param_array
        # otherwise it doesn't matter since the loop will run again anyway
    # return the orbital parameter array
    return orb_param_array


# define list of all CNEOS state vectors
cneos_sv_list = list(np.array([cneos_v, cneos_t, cneos_latitude, cneos_longitude, cneos_elevation, cneos_azim, cneos_zen]).T)
# define array to save the orbital parameters to
orb_param_array_all_cneos_fireballs = np.zeros((len(cneos_sv_list), n_monte_carlo_samples, len(orb_param_variables)))

# perform the Monte Carlo cloning and compute the orbital parameters for each cloned state vector
for i, sv in tqdm(enumerate(cneos_sv_list), total=len(cneos_sv_list)):
    # unpack
    v, t, a, o, e, azim, zen = sv
    # repack with the kde models and repeat 100 times to clone it
    repacked_sv = np.repeat([[kde_model_dv, kde_model_log_drad, v, t, a, o, e, azim, zen]], n_monte_carlo_samples, axis=0)
    # now compute orbital parameters from the state vectors for each Monte Carlo sample using multiprocessing:
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)  # avoids locking up CPU
    # convert cloned state vectors with orbital parameters
    orb_param_array_single_cneos_fireball = np.array(list(pool.imap(get_monte_carlo_orbital_parameters, repacked_sv)))
    # add the orbital parameter array for the single fireball to the array for all fireballs
    orb_param_array_all_cneos_fireballs[i] = orb_param_array_single_cneos_fireball

np.save('orb_param_array_all_cneos_fireballs_1000_clones.npy', orb_param_array_all_cneos_fireballs)  # save as a .npy file