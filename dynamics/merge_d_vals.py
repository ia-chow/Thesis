import numpy as np
import os

directory = './emccd_d_vals'
arrs = []

for filename in os.listdir(directory):
    if 'dhs' in filename:
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            # append to arrays list
            arrs.append(np.load(f))

# combine
combined_array = np.concatenate(arrs, axis=0)
# save
save = 'all_emccd_dh_vals.npy'
np.save(os.path.join(directory, save), combined_array)