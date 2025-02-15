import numpy as np
import sys
import os
import multiprocessing

# Make WMPL directory visible
sys.path.append('../../')

# Import WMPL files:
import source.WesternMeteorPyLib.wmpl.MetSim.GUI as gui
import source.WesternMeteorPyLib.wmpl.MetSim.MetSimErosion as erosion
import source.WesternMeteorPyLib.wmpl.Utils.Pickling as pickling

# Bolometric power of a zero magnitude meteor (W):
P0m = 3030  # bolometric power of zero magnitude meteor in watts (3030 for silicon bandpass CNEOS at 6000 K blackbody)
logl_scale_factor = 1.e21  # factor to scale the log likelihood by

# Modified version to run with optimization routine (now including single-body fragments):
null=None
class MetSimObj():
    def __init__(self, traj_path, const_json_file, lc_path, fixed_frag_indices, free_frag_indices, er_frag_indices):
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
        self.traj = pickling.loadPickle(*os.path.split(traj_path))
        # self.usg_data, self.traj = gui.loadUSGInputFile(*os.path.split(traj_path))
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
        if lc_path is not None:
            self.lc_data = gui.LightCurveContainer(*os.path.split(os.path.abspath(lc_path)))
        # if self.usg_data is not None:
        #     self.const.P_0m = self.usg_data.P_0m_bolo
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
        frag_numbers = []
        frag_ab_coeffs = []
        frag_gammas = []
        frag_mis = []
        # free
        frag_masses = []
        frag_er_coeffs = []
        frag_grain_mins = []
        frag_grain_maxs = []
        frag_heights = []

        # set free and fixed params
        #### CHANGE THESE LINES TO TEST DIFFERENT COMBINATIONS OF FREE PARAMETERS
        self.fixed_frag_indices = fixed_frag_indices
        self.free_frag_indices = free_frag_indices
        self.er_frag_indices = er_frag_indices
        # set up masks for free and fixed indices
        # free
        free_frag_mask = np.zeros(num_frags, bool)
        free_frag_mask[free_frag_indices] = True  # only free frags
        # fixed
        fixed_frag_mask = np.ones(num_frags, bool)
        fixed_frag_mask[free_frag_indices] = False  # everything EXCEPT free frags (i.e. fixed frags)
        # erosion
        er_frag_mask = np.zeros(num_frags, bool)
        er_frag_mask[er_frag_indices] = True  # only frags that have erosion coefficients (i.e. all free fragments excluding dust)
        
        for i, frag in enumerate(const_json.get('fragmentation_entries')):
            ### If sigma, gamma and erosiion coefficient are none, convert to whatever the default value is 
            # fixed
            frag_types.append(frag['frag_type'])
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
            if frag['erosion_coeff'] == None:  # this means that the fragment is dust or a single-body fragment
                frag_er_coeffs.append(0.)  # erosion coefficient of zero
            else:
                frag_er_coeffs.append(frag['erosion_coeff'])
            if frag['grain_mass_min'] == None:  # this means the fragment is a single-body fragment
                frag_grain_mins.append(null)  # append null
            else:
                frag_grain_mins.append(frag['grain_mass_min'])
            if frag['grain_mass_max'] == None:  # this means the fragment is a single-body fragment
                frag_grain_maxs.append(null)  # append null
            else:
                frag_grain_maxs.append(frag['grain_mass_max'])
            frag_heights.append(frag['height'])
        # print(frag_grain_mins, frag_grain_maxs)
        # fixed_frag_mask = np.argsort(fixed_frag_mask)
        # assign free fixed parameters to object
        self.free_params = [m_init, 
                            list(np.array(frag_masses)[free_frag_mask]), 
                            list(np.array(frag_er_coeffs)[er_frag_mask]),  # use er frag mask for this one! 
                            list(np.array(frag_grain_mins)[er_frag_mask]), 
                            list(np.array(frag_grain_maxs)[er_frag_mask]),
                            list(np.array(frag_heights)[free_frag_mask])  # heights
                           ]
        self.fixed_params = [dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, 
                             v_init, shape_factor, 
                            gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, 
                            erosion_bins_per_10mass, erosion_coeff, erosion_height_change, 
                            erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, 
                            erosion_rho_change, erosion_sigma_change, compressive_strength, 
                            disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, 
                            disruption_mass_min_ratio, disruption_mass_max_ratio, 
                            frag_types, frag_numbers, 
                            frag_ab_coeffs, frag_gammas, frag_mis,
                            list(np.array(frag_masses)[fixed_frag_mask]), 
                            list(np.array(frag_er_coeffs)[~er_frag_mask]),  # and inverse of er frag mask 
                            list(np.array(frag_grain_mins)[~er_frag_mask]), 
                            list(np.array(frag_grain_maxs)[~er_frag_mask]),
                            list(np.array(frag_heights)[fixed_frag_mask])
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
            m_init, frag_masses_free, frag_er_coeffs_free, frag_grain_mins_free, frag_grain_maxs_free, frag_heights_free = free_params 
            dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio, frag_types, frag_numbers, frag_ab_coeffs, frag_gammas, frag_mis, frag_masses_fixed, frag_er_coeffs_fixed, frag_grain_mins_fixed, frag_grain_maxs_fixed, frag_heights_fixed = fixed_params
            # load all the global parameters into the object
            consts = dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio
            self.loadGlobalParameters(consts)
            # combine free fixed fragmentation parameters
            frag_order = np.argsort(self.fixed_frag_indices + self.free_frag_indices)
            frag_masses = np.concatenate((frag_masses_fixed, frag_masses_free))[frag_order]
            frag_er_coeffs, frag_grain_mins, frag_grain_maxs = np.zeros((3, len(frag_masses)))  # number of fragments
            frag_er_coeffs[self.er_frag_indices] = frag_er_coeffs_free  # set all erosion fragments to their values, everything else is zero
            frag_grain_mins[self.er_frag_indices] = frag_grain_mins_free
            frag_grain_maxs[self.er_frag_indices] = frag_grain_maxs_free
            # frag_grain_mins = np.concatenate((frag_grain_mins_fixed, frag_grain_mins_free))[frag_order]
            # frag_grain_maxs = np.concatenate((frag_grain_maxs_fixed, frag_grain_maxs_free))[frag_order]
            frag_heights = np.concatenate((frag_heights_fixed, frag_heights_free))[frag_order]

            # print(frag_order, np.concatenate((frag_masses_fixed, frag_masses_free)), frag_masses)
            
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

# Novo Mesto:
event_path = '../novo_mesto/ablation_modelling/20200228_093032_trajectory_20220213_nine_stations'

FIXED_FRAG_INDICES = []
FREE_FRAG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]#, 8, 9]#, 6, 7, 8]  # fragments with free parameters
ER_FRAG_INDICES = [0, 1, 2, 4, 5, 6, 7]#, 8, 9]#, 2, 3, 4]  # free fragments that have erosion coefficients

metsim_obj = MetSimObj(traj_path=event_path + '.pickle', 
                       const_json_file=event_path + '.picklesim_fit_latest.json',
                       lc_path='../novo_mesto/ablation_modelling/light_curve_corr_edited_sorted.csv',
                       fixed_frag_indices = FIXED_FRAG_INDICES,
                       free_frag_indices = FREE_FRAG_INDICES,
                       er_frag_indices = ER_FRAG_INDICES
                      )
# initialize simulation, run with all parameters
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
def unflatten_list(lst, num_global_params=1, num_frag_params=5, er_frag_indices=ER_FRAG_INDICES):
    """
    change number of global parameters and frag parameters based on the structure of the free parameters object
    passed into the get_lc_cost_function() function
    """
    # print(lst[:num_global_params], list(zip(*[iter(lst[num_global_params:])]*(len(lst[num_global_params:])//num_frag_params))))
    num_frag_params = max(1, num_frag_params)
    sizes = list(np.repeat('k', num_frag_params - 3))  # CHANGE THIS IF STRUCTURE CHANGES
    sizes.insert(0, int(num_global_params))  # CHANGE THIS IF STRUCTURE CHANGES
    sizes[2:2] = [int(len(er_frag_indices))] * 3  # CHANGE THESE INDICES IF STRUCTURE CHANGES
    # print(sizes)
    k = (len(lst) - sum(s for s in sizes if s != 'k')) // sizes.count('k') if 'k' in sizes else 0
    result, index = [], 0
    for size in sizes:
        group_size = k if size == 'k' else size
        if group_size == 1:
            result.append(lst[index])
        else:
            result.append(lst[index:index + group_size])
        index += group_size
    return result
 
def sim_lc(flattened_free_params, metsim_obj):
    """
    Same format as function below
    """
    # compute the simulated integrated intensity for the given set of free parameters
    # unflatten the list of free parameters to pass into the initializeSimulation() function
    free_params = unflatten_list(list(flattened_free_params))
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

### define cost function for the opitmization routine

def get_lc_cost_function(flattened_free_params, metsim_obj, pso=False):
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
    # print(f'parameters: {flattened_free_params}')
    # observed LC intensity, this doesn't change
    obs_lc_intensity = P0m * (10 ** (np.concatenate(list(metsim_obj.lc_data.abs_mag_data.values()))/(-2.5)))
    # obs_lc_intensity = P0m * (10 ** (metsim_obj.usg_data.absolute_magnitudes/(-2.5)))

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
        # print('increasing height')
    else:
        first_increasing_height_arr = -1
    # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point
    simulated_lc_intensity = np.interp(np.concatenate(list(metsim_obj.lc_data.height_data.values())), 
                                       np.flip(simulation_results.leading_frag_height_arr[:first_increasing_height_arr]), 
                                       np.flip((P0m * (10 ** (simulation_results.abs_magnitude/(-2.5))))[:first_increasing_height_arr]))
    # return the negative log-likelihood (negative residual sum of square difference between the two)
    # assuming there is no error for the observed LC data
    # print(len(obs_lc_intensity), len(simulated_lc_intensity))
    log_likelihood = (-1/2 * np.nansum((obs_lc_intensity - simulated_lc_intensity) ** 2))/logl_scale_factor  # scale this
    # print(f'negative logL: {-log_likelihood}')
    # negative log lieklihood
    return -log_likelihood

# Initial guess:
initial_guess = np.load('./initial_guesses/initial_guess_novo_mesto.npy')  # initial pre-saved guess

# Print:
lamb_func = lambda free_params_flattened: get_lc_cost_function(free_params_flattened, metsim_obj=metsim_obj, pso=False)
print(f'Initial guess:{lamb_func(initial_guess)}')  # DENIS' BEST FIT FOR NOVO MESTO IS ~4.299

#### USE THIS TO TEST DIFFERENT COMBINATIONS OF TEST PARAMETERS
# Modified guess:
initial_guess_mod = initial_guess.copy()
# fiddle with global mass
initial_guess_mod[0] = initial_guess[0]
# fiddle with frag masses
initial_guess_mod = initial_guess + [0., # total mass
                                     0., 0., 0., 0., 0., 0., 0., 0.,# 0., 0., # frag mass pcts
                                     0., 0., 0., 0., 0., 0., 1.e-8,# 1.e-8, 0., # erosion coeffs
                                     0., 0., 0., 0., 0., 0., 0.,# 0.,# 1.e-4, # grain mins
                                     0., 0., 0., 0., 0., 1.e-3, 0.,# 0.,# 0.,  # grain maxs
                                     0., 1000., 0., 0., 0., 0., 0., 0.,# 0., 0., # heights
                                    ] 
print(f'Modified: {lamb_func(initial_guess_mod)}')

# "Hard" bounds for the priors:

# Generate the tuples based on the length of free fragments
fragmentation_count = len(FREE_FRAG_INDICES)
# Fragment mass percents
frag_mass_percents = tuple((0., 100.) for _ in range(fragmentation_count))
# Frag ablation coeffs
# these are 0.01 for dust and between (0.002 * 1e-6, 0.025 * 1e-6) for eroding fragments
# frag_ablation_coeffs = tuple((0.002 * 1e-6, 0.025 * 1e-6) for _ in range(fragmentation_count))
# frag_ablation_coeffs = tuple([(0.01 * 1e-6, 0.01 * 1e-6) if entry.frag_type == 'D'
#                               else (0.002 * 1e-6, 0.025 * 1e-6) for entry in metsim_obj.fragmentation_entries])
# Frag erosion coeffs
# these are fixed at zero for dust and between (0.23 * 1e-6, 6.0 * 1e-6) for eroding fragments
# frag_erosion_coeffs = tuple([(0., 0.) if entry.frag_type == 'D' 
#                              else (0.23 * 1e-6, 6.0 * 1e-6) for entry in metsim_obj.fragmentation_entries])
# keep lower bound above 0.1e6 because that's where the initial guess is
frag_erosion_coeffs = tuple([(0.01 * 1e-6, 10.0 * 1e-6) for entry in 
                             [metsim_obj.fragmentation_entries[i] for i in ER_FRAG_INDICES]])
# frag_erosion_coeffs = tuple((0.23 * 1e-6, 6.0 * 1e-6) for _ in range(fragmentation_count))
# Frag grain mins
# frag_grain_mins = tuple((1e-8, 1e-2) for _ in range(fragmentation_count))
frag_grain_mins = tuple([(5e-6, 1e-3) for entry in [metsim_obj.fragmentation_entries[i] for i in ER_FRAG_INDICES]])
# Frag grain maxs
frag_grain_maxs = tuple([(1e-4, 1e-2) for entry in [metsim_obj.fragmentation_entries[i] for i in ER_FRAG_INDICES]])
# Frag heights
frag_heights = tuple((20., 80.) for _ in range(fragmentation_count))
# Combine them into a single flat tuple
#### CHANGE THE RESULTS/BOUNDS DEPENDING ON WHAT COMBINATION IS BEING USED!
result = frag_mass_percents + frag_erosion_coeffs + frag_grain_mins + frag_grain_maxs + frag_heights
# Print the result to check the output
hard_bounds = np.array(((1.e2, 5.e4), ) + result)


# Least restrictive hard bounds
bounds = hard_bounds

# # Optionally include more restrictive bounds for the priors:
# # initialize particles
# cov_factor = 1.e-1  # CHANGE THIS DEPENDING ON THE EVENT
# cov = np.float64(np.diag(np.ones(len(initial_guess)) * (initial_guess ** 2))) * cov_factor
# # cov = np.eye(len(initial_guess), len(initial_guess)) * cov_factor
# mu = initial_guess
# bounds = tuple([tuple(np.clip((mu[i] - 2. * np.diag(np.sqrt(cov))[i], mu[i] + 2. * np.diag(np.sqrt(cov))[i]), 
#                         hard_bounds[i][0], hard_bounds[i][1])) for i in range(0, len(mu))])

# manually change certain bounds
bounds = np.array(bounds)
# mass bounds
bounds[0] = (1.e2, 5.e4)
bounds[1:len(frag_mass_percents) + 1] = np.repeat([(0., 100.)], len(frag_mass_percents), axis=0)
# Height bounds:
# CHANGE THIS DEPENDING ON HEIGHT
height_bounds = 850.  # permissible values plus or minus the starting fragmentation height
bounds[-fragmentation_count:] = np.array((initial_guess[-fragmentation_count:] - height_bounds, 
                                                         initial_guess[-fragmentation_count:] + height_bounds)).T
bounds = tuple([tuple(bound) for bound in bounds])
print(bounds)

# prior and log-likelihood
import signal

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

# PRIOR
def prior(cube, tup_bounds=bounds, n_frags=fragmentation_count, er_frags=len(ER_FRAG_INDICES)):
    """
    Transform the unit cube to a uniform prior
    """
    # bounds
    bounds = np.array(tup_bounds)
    mass_bnds = bounds[0]
    frag_mass_pct_bnds = bounds[1:n_frags + 1]
    frag_er_coeff_bnds = bounds[n_frags + 1: n_frags + er_frags + 1]
    frag_grain_min_bnds = bounds[n_frags + er_frags + 1: n_frags + 2 * er_frags + 1]
    frag_grain_max_bnds = bounds[n_frags + 2 * er_frags + 1: n_frags + 3 * er_frags + 1]
    frag_height_bnds = bounds[n_frags + 3 * er_frags + 1: 2 * n_frags + 3 * er_frags + 1]
    # particle params
    # mass
    cube[0] = cube[0] * (mass_bnds[1] - mass_bnds[0]) + mass_bnds[0] # transform [0, 1] unit cube to [mass_lower, mass_uper]
    # mass pct
    cube[1:n_frags + 1] = cube[1:n_frags + 1] * (frag_mass_pct_bnds[:, 1] - frag_mass_pct_bnds[:, 0]) + frag_mass_pct_bnds[:, 0]  # transform [0, 1] unit cube
    # erosion coeff
    cube[n_frags + 1: n_frags + er_frags + 1] = cube[n_frags + 1: n_frags + er_frags + 1] * (frag_er_coeff_bnds[:, 1] - frag_er_coeff_bnds[:, 0]) + frag_er_coeff_bnds[:, 0]
    # grain mins
    cube[n_frags + er_frags + 1: n_frags + 2 * er_frags + 1] = cube[n_frags + er_frags + 1: n_frags + 2 * er_frags + 1] * (frag_grain_min_bnds[:, 1] - frag_grain_min_bnds[:, 0]) + frag_grain_min_bnds[:, 0]
    # grain maxs
    cube[n_frags + 2 * er_frags + 1: n_frags + 3 * er_frags + 1] = cube[n_frags + 2 * er_frags + 1: n_frags + 3 * er_frags + 1] * (frag_grain_max_bnds[:, 1] - cube[n_frags + er_frags + 1: n_frags + 2 * er_frags + 1]) + cube[n_frags + er_frags + 1: n_frags + 2 * er_frags + 1]
    # heights
    cube[n_frags + 3 * er_frags + 1: 2 * n_frags + 3 * er_frags + 1] = cube[n_frags + 3 * er_frags + 1: 2 * n_frags + 3 * er_frags + 1] * (frag_height_bnds[:, 1] - frag_height_bnds[:, 0]) + frag_height_bnds[:, 0]
    return cube

def log_likelihood(particle, metsim_obj=metsim_obj, timeout=15):
    """
    similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
    """
    # observed LC intensity, this doesn't change
    obs_lc_intensity = P0m * (10 ** (np.concatenate(list(metsim_obj.lc_data.abs_mag_data.values()))/(-2.5)))
    # obs_lc_intensity = P0m * (10 ** (metsim_obj.usg_data.absolute_magnitudes/(-2.5)))

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the timer for timeout
    # get simulated LC intensity onthe object
    try: 
        simulation_results = sim_lc(particle, metsim_obj)
    except TimeoutException:
        print('timeout')
        return -np.inf  # immediately return -np.inf if times out
    finally:
        signal.alarm(0)  # Cancel alarm
    # compute simulated LC intensity
    # find where height starts increasing if it does at any point
    # print(np.diff(simulation_results.leading_frag_height_arr))
    if np.sum(np.diff(simulation_results.leading_frag_height_arr) > 0) > 0:
        first_increasing_height_arr = np.where(np.diff(simulation_results.leading_frag_height_arr) > 0)[0][0] - 1
        # print('increasing height')
    else:
        first_increasing_height_arr = -1
    # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point
    simulated_lc_intensity = np.interp(np.concatenate(list(metsim_obj.lc_data.height_data.values())), 
                                       np.flip(simulation_results.leading_frag_height_arr[:first_increasing_height_arr]), 
                                       np.flip((P0m * (10 ** (simulation_results.abs_magnitude/(-2.5))))[:first_increasing_height_arr]))
    # return the log-likelihood (residual sum of square difference between the two)
    # assuming there is no error for the observed LC data
    # print(len(obs_lc_intensity), len(simulated_lc_intensity))
    log_likelihood = (-1/2 * np.nansum((obs_lc_intensity - simulated_lc_intensity) ** 2))/logl_scale_factor  # scale this
    # print(f'negative logL: {-log_likelihood}')
    # positive log lieklihood unlike get_lc_cost_function
    return log_likelihood

# Set up NS:
n_params = len(initial_guess)

# Run NS:
import dynesty
filename = './dynesty_saves/novo_mesto_unrestricted_sorted.save'  # CHANGE THIS

n_threads = multiprocessing.cpu_count() - 1
# n_threads = 32
# n_threads = 48
# n_threads = 50
timeout = 15  # number of seconds to timeout

with dynesty.pool.Pool(n_threads, log_likelihood, prior, 
                       logl_args=(metsim_obj, timeout), 
                       ptform_args=(bounds, fragmentation_count, len(ER_FRAG_INDICES))) as pool:
    # NEW RUN
    dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, n_params, nlive=1000, pool = pool)
    dsampler.run_nested(print_progress=True, checkpoint_file=filename)

    # # RESUME:
    # dsampler = dynesty.DynamicNestedSampler.restore(filename, pool = pool)
    # dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=filename)

print('done')