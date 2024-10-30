import numpy as np
import pandas as pd
import sys
import json
import os
import copy
import matplotlib.pyplot as plt
import scipy

# Make WMPL directory visible

sys.path.append('../../')

# Import WMPL files

import source.WesternMeteorPyLib.wmpl.MetSim.MetSim as metsim
import source.WesternMeteorPyLib.wmpl.MetSim.FitSim as fitsim
import source.WesternMeteorPyLib.wmpl.MetSim.GUI as gui
import source.WesternMeteorPyLib.wmpl.MetSim.MetSimErosion as erosion

# MetSimObj() object, modified version of the MetSimGUI() object in GUI.py to run with the optimization routine

class MetSimObj():
    def __init__(self, traj_path, const_json_file, fixed_frag_indices, free_frag_indices):
        # Init an axis for the electron line density
        # self.electronDensityPlot = self.magnitudePlot.canvas.axes.twiny()
        self.electron_density_plot_show = False
        ### Wake parameters ###
        self.wake_on = False
        self.wake_show_mass_bins = False
        self.wake_ht_current_index = 0
        self.current_wake_container = None
        # if self.wake_heights is not None:
        #     self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]
        # else:
        #     self.wake_plot_ht = self.traj.rbeg_ele # m
        self.wake_normalization_method = 'area'
        self.wake_align_method = 'none'
        self.magnitudePlotWakeLines = None
        self.magnitudePlotWakeLineLabels = None
        self.velocityPlotWakeLines = None
        self.lagPlotWakeLines = None
        self.usg_data, self.traj = gui.loadUSGInputFile(*os.path.split(traj_path))
        self.dir_path = os.path.dirname(traj_path)
        # Disable different density after erosion change
        self.erosion_different_rho = False
        # Disable different ablation coeff after erosion change
        self.erosion_different_sigma = False
        # Disable different erosion coeff after disruption at the beginning
        self.disruption_different_erosion_coeff = False
        # Fragmentation object
        self.fragmentation = None
        self.simulation_results = None
        self.const_prev = None
        self.simulation_results_prev = None
        self.const = erosion.Constants()  # initialize this, these will be replaced later
        self.const.P_0m = self.usg_data.P_0m_bolo
        # If a JSON file with constant was given, load them instead of initing from scratch
        if const_json_file is not None:
            # Load the constants from the JSON files
            self.const, const_json = gui.loadConstants(const_json_file)
            # Init the fragmentation container for the GUI
            if len(self.const.fragmentation_entries):
                self.fragmentation = gui.FragmentationContainer(self, \
                    os.path.join(self.dir_path, self.const.fragmentation_file_name))
                self.fragmentation.fragmentation_entries = self.const.fragmentation_entries
                # Overwrite the existing fragmentatinon file
                # self.fragmentation.writeFragmentationFile()
            # Check if the disruption erosion coefficient is different than the main erosion coeff
            if const_json['disruption_erosion_coeff'] != const_json['erosion_coeff']:
                self.disruption_different_erosion_coeff = True
            # Check if the density is changed after Hchange
            if 'erosion_rho_change' in const_json:
                if const_json['erosion_rho_change'] != const_json['rho']:
                    self.erosion_different_rho = True
            # Check if the ablation coeff is changed after Hchange
            if 'erosion_sigma_change' in const_json:
                if const_json['erosion_sigma_change'] != const_json['sigma']:
                    self.erosion_different_sigma = True
        else:
            raise('no json file!')

        ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

        # Determine the height range for fitting the density
        self.dens_fit_ht_beg = self.const.h_init
        self.dens_fit_ht_end = self.traj.rend_ele - 5000
        if self.dens_fit_ht_end < 14000:
            self.dens_fit_ht_end = 14000

        # Fit the polynomail describing the density
        dens_co = gui.MetSimGUI.fitAtmosphereDensity(self, self.dens_fit_ht_beg, self.dens_fit_ht_end)
        self.const.dens_co = dens_co

        # get global parameters from json file, everything other than params marked "free" are fixed
        dt = const_json.get('dt')
        P_0m = const_json.get('P_0m')
        h_init = const_json.get('h_init')
        m_kill = const_json.get('m_kill')
        v_kill = const_json.get('v_kill')
        h_kill = const_json.get('h_kill')
        len_kill = const_json.get('len_kill') 
        rho = const_json.get('rho')  # free
        rho_grain = const_json.get('rho_grain')  # free
        m_init = const_json.get('m_init')  # free
        sigma = const_json.get('sigma')  # free
        v_init = const_json.get('v_init')
        shape_factor = const_json.get('shape_factor')
        gamma = const_json.get('gamma')
        zenith_angle = const_json.get('zenith_angle')
        lum_eff = const_json.get('lum_eff')
        lum_eff_type = const_json.get('lum_eff_type')
        erosion_height_start = const_json.get('erosion_height_start')
        erosion_bins_per_10mass = const_json.get('erosion_bins_per_10mass')
        erosion_coeff = const_json.get('erosion_coeff')
        erosion_height_change = const_json.get('erosion_height_change')
        erosion_coeff_change = const_json.get('erosion_coeff_change')
        erosion_mass_index = const_json.get('erosion_mass_index')
        erosion_mass_min = const_json.get('erosion_mass_min')
        erosion_mass_max = const_json.get('erosion_mass_max')
        erosion_rho_change = const_json.get('rho')
        erosion_sigma_change = const_json.get('sigma')
        compressive_strength = const_json.get('compressive_strength')
        disruption_erosion_coeff = const_json.get('erosion_coeff')
        disruption_mass_grain_ratio = const_json.get('disruption_mass_grain_ratio')
        disruption_mass_index = const_json.get('disruption_mass_index')
        disruption_mass_min_ratio = const_json.get('disruption_mass_min_ratio')
        disruption_mass_max_ratio = const_json.get('disruption_mass_max_ratio')

        # get fragmentation parameters from json file
        # type, height, number, gamma, mass index are fixed, 
        # mass, ablation coefficient, erosion coefficient, grain min, grain max are not fixed
        num_frags = len(const_json.get('fragmentation_entries'))
        # fixed
        frag_types = []
        frag_heights = []
        frag_numbers = []
        frag_ab_coeffs = []
        frag_gammas = []
        frag_mis = []
        # free
        frag_masses = []
        frag_er_coeffs = []
        frag_grain_mins = []
        frag_grain_maxs = []

        for i, frag in enumerate(const_json.get('fragmentation_entries')):
            ### If sigma, gamma and erosiion coefficient are none, convert to whatever the default value is 
            # fixed
            frag_types.append(frag['frag_type'])
            frag_heights.append(frag['height'])
            frag_numbers.append(frag['number'])
            if frag['gamma'] == None:
                frag_gammas.append(gamma)
            else:
                frag_gammas.append(frag['gamma'])
            frag_mis.append(frag['mass_index'])
            if frag['sigma'] == None: 
                frag_ab_coeffs.append(sigma) 
            else:
                frag_ab_coeffs.append(frag['sigma'])
            # free
            frag_masses.append(frag['mass_percent'])
            if frag['erosion_coeff'] == None:
                frag_er_coeffs.append(0.)  # erosion coefficient of zero
            else:
                frag_er_coeffs.append(frag['erosion_coeff'])
            frag_grain_mins.append(frag['grain_mass_min'])
            frag_grain_maxs.append(frag['grain_mass_max'])
            
        # set free and fixed params
        #### CHANGE THESE LINES TO TEST DIFFERENT COMBINATIONS OF FREE PARAMETERS
        self.fixed_frag_indices = fixed_frag_indices
        self.free_frag_indices = free_frag_indices
        # set up masks for free and fixed indices
        free_frag_mask = np.zeros(num_frags, bool)
        free_frag_mask[free_frag_indices] = True  # only free frags
        fixed_frag_mask = np.ones(num_frags, bool)
        fixed_frag_mask[free_frag_indices] = False  # everything EXCEPT free frags (i.e. fixed frags)
        # assign free fixed parameters to object
        self.free_params = [m_init, 
                            list(np.array(frag_masses)[free_frag_mask]), 
                            list(np.array(frag_er_coeffs)[free_frag_mask]), 
                            list(np.array(frag_grain_mins)[free_frag_mask]), 
                            list(np.array(frag_grain_maxs)[free_frag_mask])
                           ]
        self.fixed_params = [dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, 
                             v_init, shape_factor, 
                            gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, 
                            erosion_bins_per_10mass, erosion_coeff, erosion_height_change, 
                            erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, 
                            erosion_rho_change, erosion_sigma_change, compressive_strength, 
                            disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, 
                            disruption_mass_min_ratio, disruption_mass_max_ratio, 
                            frag_types, frag_heights, frag_numbers, 
                            frag_ab_coeffs, frag_gammas, frag_mis,
                            list(np.array(frag_masses)[fixed_frag_indices]), 
                            list(np.array(frag_er_coeffs)[fixed_frag_indices]), 
                            list(np.array(frag_grain_mins)[fixed_frag_indices]), 
                            list(np.array(frag_grain_maxs)[fixed_frag_indices])
                            ]
        
        # load all the global parameters into the object
        consts = dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio
        self.loadGlobalParameters(consts)

        # get all parameters
        self.all_params = (self.free_params, self.fixed_params)
        
        # self.initializeSimulation(all_params)
        # self.initializeSimulation(const_json)
    
    def loadGlobalParameters(self, consts):
            """
            Loads the global parameters (constants) into the object
            """
            dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio = consts
            # load all the non-fragmentation parameters into the object
            # 33 parameters
            self.const.dt = dt
            self.const.P_0m = P_0m
            self.const.h_init = h_init
            self.const.m_kill = m_kill
            self.const.v_kill = v_kill
            self.const.h_kill = h_kill
            self.const.len_kill = len_kill
            self.const.rho = rho
            self.const.rho_grain = rho_grain
            self.const.m_init = m_init
            self.const.sigma = sigma
            self.const.v_init = v_init
            self.const.shape_factor = shape_factor
            self.const.gamma = gamma
            self.const.zenith_angle = zenith_angle
            self.const.lum_eff = lum_eff
            self.const.lum_eff_type = lum_eff_type
            self.const.erosion_height_start = erosion_height_start
            self.const.erosion_bins_per_10mass = erosion_bins_per_10mass
            self.const.erosion_coeff = erosion_coeff
            self.const.erosion_height_change = erosion_height_change
            self.const.erosion_coeff_change = erosion_coeff_change
            self.const.erosion_mass_index = erosion_mass_index
            self.const.erosion_mass_min = erosion_mass_min
            self.const.erosion_mass_max = erosion_mass_max
            self.const.erosion_rho_change = erosion_rho_change
            self.const.erosion_sigma_change = erosion_sigma_change
            self.const.compressive_strength = compressive_strength
            self.const.disruption_erosion_coeff = disruption_erosion_coeff
            self.const.disruption_mass_grain_ratio = disruption_mass_grain_ratio
            self.const.disruption_mass_index = disruption_mass_index
            self.const.disruption_mass_min_ratio = disruption_mass_min_ratio
            self.const.disruption_mass_max_ratio = disruption_mass_max_ratio

    def initializeSimulation(self, all_params):
            """ Run the simulation and show the results. """
            # If the fragmentation is turned on and no fragmentation data is given, notify the user
            # if self.const.fragmentation_on and (self.fragmentation is None):
            #     frag_error_message = QMessageBox(QMessageBox.Critical, "Fragmentation file error", \
            #         "Fragmentation is enabled but no fragmentation file is set.")
            #     frag_error_message.setInformativeText("Either load an existing fragmentation file or create a new one.")
            #     frag_error_message.exec_()
            #     return None   

            # unpack all params
            free_params, fixed_params = all_params
            # unpack again
            #### CHANGE THESE NEXT TWO LINES TO TEST DIFFERENT COMBINATIONS OF FREE PARAMETERS
            m_init, frag_masses_free, frag_er_coeffs_free, frag_grain_mins_free, frag_grain_maxs_free = free_params 
            dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio, frag_types, frag_heights, frag_numbers, frag_ab_coeffs, frag_gammas, frag_mis, frag_masses_fixed, frag_er_coeffs_fixed, frag_grain_mins_fixed, frag_grain_maxs_fixed = fixed_params
            # load all the global parameters into the object
            consts = dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio
            self.loadGlobalParameters(consts)
            # combine free fixed fragmentation parameters
            frag_order = self.fixed_frag_indices + self.free_frag_indices
            frag_masses = np.concatenate((frag_masses_fixed, frag_masses_free))[frag_order]
            frag_er_coeffs = np.concatenate((frag_er_coeffs_fixed, frag_er_coeffs_free))[frag_order]
            frag_grain_mins = np.concatenate((frag_grain_mins_fixed, frag_grain_mins_free))[frag_order]
            frag_grain_maxs = np.concatenate((frag_grain_maxs_fixed, frag_grain_maxs_free))[frag_order]
            
            # Load fragmentation entries
            self.fragmentation_entries = []
            for i in range(0, len(frag_masses)):  # pick any frag entry to iterate over
                frag_entry = gui.FragmentationEntry(frag_types[i], frag_heights[i], frag_numbers[i], frag_masses[i], 
                                                    frag_ab_coeffs[i], frag_gammas[i], frag_er_coeffs[i], frag_grain_mins[i], frag_grain_maxs[i], frag_mis[i])
                self.fragmentation_entries.append(frag_entry)
            # set the fragmentation entries to constants
            self.const.fragmentation_entries = self.fragmentation_entries
    
            # Sort entries by height
            self.fragmentation.sortByHeight()

            # Reset the status of all fragmentations
            self.fragmentation.resetAll()
        
            # fragmentation
            self.const.fragmentation_on = True

            # print(self.const.fragmentation_on)
    
            # Run the simulation
            frag_main, results_list, wake_results = erosion.runSimulation(self.const, compute_wake=self.wake_on)
            # print(results_list)

            # Store simulation results
            self.simulation_results = gui.SimulationResults(self.const, frag_main, results_list, wake_results)

# Test using the following event and fragmentation indices (change as needed)

FIXED_FRAG_INDICES = [0, 1, 2, 3, 4, 5, 6, 8]
FREE_FRAG_INDICES = [7]  # fragments with free parameters

metsim_obj = MetSimObj(traj_path='../usg_metsim_files/1994-02-01/usg_input_feb_1994.txt', 
                       const_json_file='../usg_metsim_files/1994-02-01/usg_input_feb_1994_sim_fit_latest.json',
                       fixed_frag_indices = FIXED_FRAG_INDICES,
                       free_frag_indices = FREE_FRAG_INDICES
                      )

# Generate a simulated LC using the "best-fit parameters" determined by analyst
metsim_obj.initializeSimulation(metsim_obj.all_params)

# Optimization:

# helper nested function to flatten a list
def flatten_list(nested_list):
    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)

    flat_list = []
    flatten(nested_list)
    return flat_list

# helper function to unflatten the list
#### CHANGE THE NUMBER OF GLOBAL/FRAGMENT PARAMETERS TO TEST DIFFERENT COMBINATIONS
def unflatten_list(lst, num_global_params=1, num_frag_params=4):
    """
    change number of global parameters and frag parameters based on the structure of the free parameters object
    passed into the get_lc_cost_function() function
    """
    # print(lst[:num_global_params], list(zip(*[iter(lst[num_global_params:])]*(len(lst[num_global_params:])//num_frag_params))))
    num_frag_params = max(1, num_frag_params)
    return list(lst[:num_global_params]) + list(zip(*[iter(lst[num_global_params:])]*(len(lst[num_global_params:])//num_frag_params)))


# Simulate a light curve given a set of free parameters and a MetsimObj() object
def sim_lc(flattened_free_params, metsim_obj):
    """
    Same format as function below
    """
    # compute the simulated integrated intensity for the given set of free parameters
    # unflatten the list of free parameters to pass into the initializeSimulation() function
    free_params = unflatten_list(flattened_free_params)
    # construct all params to pass into intiializeSimulation function
    # print(free_params)
    all_params = (free_params, metsim_obj.fixed_params)  # set of all params to use
    # run simulation on the object, copying it
    # obj_copy = copy.copy(metsim_obj)
    # obj_copy.initializeSimulation(all_params)
    metsim_obj.initializeSimulation(all_params)
    # return the simulation results
    # return obj_copy.simulation_results
    return metsim_obj.simulation_results


### define cost function for the optimization routine
def get_lc_cost_function(flattened_free_params, metsim_obj, pso):
    """
    Gets the residuals for the integrated intensity given
    a metsim object and a list/tuple of an initial guess of free parameters
    
    metsim_obj is an MetSim() object
    flattened_free_params is an initial guess flattened list of free parameters with the form
    flattened_free_params = [rho, rho_grain, m_init, sigma, *frag_masses, 
    *frag_ab_coeffs, *frag_er_coeffs, *frag_grain_mins, *frag_grain_maxs]
    pso is a boolean for if pso is being used

    Global parameters:
    rho: float
    rho_grain: float
    m_init: float
    sigma: float

    Fragment parameters (for n total fragments):
    *frag_masses: n fragment masses
    *frag_ab_coeffs: n ablation coefficients
    *frag_er_coeffs: n - nd erosion coefficients where nd is the number of dust fragments
    (as dust fragments do not have a nassociated erosion coefficient)
    *frag_grain_mins: n grain mins
    *frag_grain_maxs: n grain maxs
    """
    # observed LC intensity, this doesn't change
    obs_lc_intensity = 3030 * (10 ** (metsim_obj.usg_data.absolute_magnitudes/(-2.5)))

    # get simulated LC intensity onthe object
    # flatten if pso is being used
    if pso:
        flattened_free_params = flattened_free_params.flatten()
    simulation_results = sim_lc(flattened_free_params, metsim_obj)
    # compute simulated LC intensity
    # find where height starts increasing if it does at any point
    # print(np.diff(simulation_results.leading_frag_height_arr))
    if np.sum(np.diff(simulation_results.leading_frag_height_arr) > 0) > 0:
        first_increasing_height_arr = np.where(np.diff(simulation_results.leading_frag_height_arr) > 0)[0][0] - 1
        print('increasing height')
    else:
        first_increasing_height_arr = -1
    # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point
    simulated_lc_intensity = np.interp(metsim_obj.traj.observations[0].model_ht, 
                                       np.flip(simulation_results.leading_frag_height_arr[:first_increasing_height_arr]), 
                                       np.flip(simulation_results.luminosity_arr[:first_increasing_height_arr]))
    # return the negative log-likelihood (negative residual sum of square difference between the two)
    # assuming there is no error for the observed LC data
    log_likelihood = (-1/2 * np.nansum((obs_lc_intensity - simulated_lc_intensity) ** 2))/1e24  # scale this
    # print(f'negative logL: {-log_likelihood}')
    # negative log lieklihood
    return -log_likelihood


# Get the initial guess from the analyzer fit 
initial_guess = np.array(flatten_list(metsim_obj.free_params))

#### USE THIS TO TEST DIFFERENT COMBINATIONS OF TEST PARAMETERS
#### Change the shape of initial_guess_mod as needed
initial_guess_mod = initial_guess.copy()
# fiddle with global mass
initial_guess_mod[0] = initial_guess[0] + 0.
# fiddle with frag masses
initial_guess_mod = initial_guess + [0., # total mass
                                     0.,  # frag mass pcts
                                     1.e-7, # erosion coeffs
                                     0., # grain mins
                                     0.  # grain maxs
                                    ] 

# Bounds for optimization:

# Generate the tuples based on the length of free fragments
fragmentation_count = len(FREE_FRAG_INDICES)
# Fragment mass percents
frag_mass_percents = tuple((0., 100.) for _ in range(fragmentation_count))
# Frag erosion coeffs
frag_erosion_coeffs = tuple([(0., 0.) if entry.frag_type == 'D' 
                             else (0.01 * 1e-6, 6.0 * 1e-6) for entry in 
                             [metsim_obj.fragmentation_entries[i] for i in FREE_FRAG_INDICES]])
# Frag grain mins
frag_grain_mins = tuple((1e-4, 1e1) for _ in range(fragmentation_count))
# Frag grain maxs
frag_grain_maxs = tuple((1e-3, 1e2) for _ in range(fragmentation_count))
# Combine them into a single flat tuple
#### CHANGE THE RESULTS/BOUNDS DEPENDING ON WHAT COMBINATION IS BEING USED!
result = frag_mass_percents + frag_erosion_coeffs + frag_grain_mins + frag_grain_maxs
# Print the result to check the output
bounds = ((1.e4, 1.e7), ) + result

# Constraints for optimization:
#### ADD MASS CONSTRAINTS IF GRAIN MIN/MAX IS BEING USED IN THE COMBINATIONS!
# generate mass constraints
mass_constraints = []
# for each fragment, impose constraint that the fragment grain min (position -2) must be less than the fragment grain max (position -1)
for i in range(len(unflatten_list(initial_guess)[-1])):
    mass_constraints.append({'type': 'eq', 
                             "fun": lambda free_params_flattened: 
                             int(unflatten_list(free_params_flattened)[-2][i] > unflatten_list(free_params_flattened)[-1][i])})

# Run normal optimization:
best_fit = scipy.optimize.minimize(fun=lambda free_params_flattened: 
                                   get_lc_cost_function(free_params_flattened, metsim_obj=metsim_obj, pso=False),
                                   x0=initial_guess_mod, 
                                   method='trust-constr',
                                   # method='L-BFGS-B',
                                   # method='SLSQP',
                                   # method='BFGS',
                                   # method='Nelder-Mead',
                                   bounds=bounds, 
                                   options={'verbose': 2, 
                                            'maxiter': 50, 
                                            # 'gtol': 9e-1
                                            # 'return_all': True,
                                           },
                                   constraints=mass_constraints, 
                                   # callback=MinimizeStopper()
                                  )#, options={'maxiter': 1e5})

# Compare original LC to simulated LCs with the initial guess and the best-fit guess:
results_initial = sim_lc(initial_guess, metsim_obj)
results_bf = sim_lc(best_fit.x, metsim_obj)

# Plot:
plt.figure(figsize = (12, 8))
# interpolate the results initial and results leading frag heights
if np.sum(np.diff(results_initial.leading_frag_height_arr) > 0) > 0:
    first_increasing_height_arr = np.where(np.diff(results_initial.leading_frag_height_arr) > 0)[0][0] - 1
else:
    first_increasing_height_arr = -1
interpolated_initial_int = np.interp(metsim_obj.traj.observations[0].model_ht, 
          np.flip(results_initial.leading_frag_height_arr[:first_increasing_height_arr]), 
          np.flip(results_initial.luminosity_arr[:first_increasing_height_arr]))
interpolated_bf_int = np.interp(metsim_obj.traj.observations[0].model_ht, 
          np.flip(results_bf.leading_frag_height_arr[:first_increasing_height_arr]), 
          np.flip(results_bf.luminosity_arr[:first_increasing_height_arr]))
# plot
plt.plot(interpolated_initial_int, metsim_obj.traj.observations[0].model_ht/1000., label = 'leading initial guess')
plt.plot(interpolated_bf_int, metsim_obj.traj.observations[0].model_ht/1000., label = 'leading best fit')
# y limit
plt.ylim(10., 80.)
plt.legend()
# show
plt.show()

#### PSO OPTIMIZATION:
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

def get_lc_cost_function_pso(free_params_flattened, metsim_obj=metsim_obj, pso=True):
    return get_lc_cost_function(free_params_flattened, metsim_obj, pso)

### Only fit parameters of fragmentation processes which are used ###
# bounds = []
param_string = 'm'

pso_iterations = 10
pso_particles = 100

# # Set up hyperparameters
# options = {'c1': 0.6, 'c2': 0.3, 'w': 0.9}
options = {'c1': 0.6, 'c2': 0.3, 'w': 0.9, 'k': 10, 'p': 1}
bounds_pso = tuple(np.array(bounds).T)
initial_guess_pso = initial_guess.reshape(1, len(initial_guess))

# generate particles in a gaussian ball around the initial guess
cov_factor = 1e-3

cov = np.diag(np.ones(len(initial_guess_pso.T)) * (initial_guess ** 2))
rng = np.random.default_rng(seed=1234)
init_pos = rng.multivariate_normal(initial_guess_pso.flatten(), cov * cov_factor, size = pso_particles)

# check cost function on all the initial particles
for pos in init_pos:
    print(get_lc_cost_function_pso(pos))

# ### HOW METSIM DOES IT:
# # Normalize the fit parameters to the [0, 1] range
# mini_norm_handle = gui.MinimizationParameterNormalization(initial_guess)
# p0_normed, bounds_normed = mini_norm_handle.normalizeBounds(bounds)

# # Set up bounds (min, max) are (0, 1)
# pso_bounds = (np.zeros(len(p0_normed)), np.ones(len(p0_normed)))

# # Create particles in a tight Gaussian around the initial parameters
# init_pos = np.random.normal(loc=p0_normed, scale=0.2 + np.zeros_like(p0_normed), 
#                             size=(pso_particles - 1, len(p0_normed)))
# init_pos[init_pos < 0] = abs(init_pos[init_pos < 0])
# init_pos[init_pos > 1] = 1 - init_pos[init_pos > 1] + 1

# # Add manual fit to initial positions
# init_pos = np.append(init_pos, np.array([p0_normed]), axis=0)

# Run PSO:
optimizer = ps.single.GlobalBestPSO(n_particles=pso_particles, 
                                    dimensions=len(initial_guess), 
                                    options=options, 
                                    bounds=bounds_pso, 
                                    init_pos=init_pos
                                   )

cost, pos = optimizer.optimize(get_lc_cost_function_pso, 
                               iters=1000, 
                               n_processes = multiprocessing.cpu_count() - 1
                              )

