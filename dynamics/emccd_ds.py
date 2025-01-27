from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing
import itertools

# import Dcriteria function
from Dcriteria import calcDSH
# Read in EMCCD data
emccd_meteors_raw = pd.read_json('solution_table.json')
# filter showers so only sporadics:
emccd_sporadic_mask = emccd_meteors_raw.shower_no == -1  # compute this since we also need to filter impact probabilities by this
emccd_meteors_sporadic = emccd_meteors_raw[emccd_sporadic_mask]  # not associated with a shower
# get raw orbital elements:
# for EMCCD fireballs
emccd_as_raw, emccd_es_raw, emccd_is_raw, emccd_omegas_raw, emccd_nodes_raw, emccd_ms_raw, emccd_qs_raw = np.array(emccd_meteors_sporadic[['a', 'e', 'i', 'omega', 'asc_node', 'mean_anomaly', 'q']]).T
# select asteroidal meteors based on the definition of borovicka et al. 2022
sma_jupiter = 5.20336301
tisserand_cutoff = 'borovicka'  # either a number to filter by tisserand parameter or 'borovicka' to use the borovicka criteria
# tisserand_cutoff = 3.0

#### EMCCD:
emccd_Qs_raw = emccd_as_raw * (1. + (emccd_es_raw))
emccd_tjs_raw = (sma_jupiter/emccd_as_raw) + 2. * np.cos(np.deg2rad(emccd_is_raw)) * np.sqrt((emccd_as_raw/sma_jupiter) * (1. - emccd_es_raw ** 2))
# mask based on criteria of borovicka et al 2022
emccd_tisserand_mask = (emccd_Qs_raw.T < 4.9) | (emccd_is_raw.T > 40.) | (emccd_es_raw.T > 0.9) if tisserand_cutoff == 'borovicka' else (emccd_tjs_raw > tisserand_cutoff) | (emccd_Qs_raw.T < 4.9)
# filter by the mask
emccd_meteors = emccd_meteors_sporadic[emccd_tisserand_mask]
# get orbital parameters of filtered objects
emccd_as, emccd_es, emccd_is, emccd_omegas, emccd_nodes, emccd_ms, emccd_qs = np.array(emccd_meteors[['a', 'e', 'i', 'omega', 'asc_node', 'mean_anomaly', 'q']]).T

# 101631 meteors
# assemble in the format taken by the calcDSH() function:
emccd_elems = np.c_[emccd_qs, emccd_es, emccd_is, emccd_nodes, emccd_omegas]

# wrapper for calcDSH
def get_DSH(i1, i2, elems=emccd_elems):
    """
    Get the dsh value for two indices
    """
    return calcDSH(*elems[i1], *elems[i2])

# n threads
n_threads = multiprocessing.cpu_count() - 1
# get D criteria for every combination of two meteors in the EMCCD data, splitting into chunks to avoid crashing:
def split_into_groups(n, group_size):
    """
    Splits n into groups of approximately equal sizes.
    """
    groups = []
    for i in range(0, n, group_size):
        end = min(i + group_size - 1, n - 1)
        groups.append((i, end))
    return groups

# set the total number of elements and group size:
n = len(emccd_elems)
group_size = 30000  # set group size
# Split into groups
groups = split_into_groups(n, group_size)

# Process within each group:
for start, end in groups:
    print(f"Processing group: {start} to {end}")
    # use multiprocessing to compute for this group:
    with multiprocessing.Pool(n_threads) as pool:
        # end + 1 because the end isn't indexed otherwise
        results = np.array(list(tqdm(pool.starmap(get_DSH, itertools.combinations(range(start, end + 1), r = 2)))))
        # save to numpy array
        np.save(f'./emccd_d_vals/emccd_ds_{start}_{end}.npy', results)

# Process cross-group pairs
for group_i, group_j in itertools.combinations(groups, 2):  # All unique group pairs
    print(f"Processing cross-group pairs: {group_i} and {group_j}")
    start_i, end_i = group_i
    start_j, end_j = group_j
    with multiprocessing.Pool(n_threads) as pool:
        results = np.array(list(tqdm(pool.starmap(get_DSH, itertools.product(range(start_i, end_i + 1), range(start_j, end_j + 1))))))
        # save to numpy array
        np.save(f'./emccd_d_vals/emccd_ds_cg_{end_i}_{end_j}.npy', results)