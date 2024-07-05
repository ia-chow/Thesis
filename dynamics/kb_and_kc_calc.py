import numpy as np
import pandas as pd
import os
import json

from wmpl.PythonNRLMSISE00.nrlmsise_00_header import (
    nrlmsise_output,
    nrlmsise_flags,
    nrlmsise_input,
    lstCalc,
)
from wmpl.PythonNRLMSISE00.nrlmsise_00 import gtd7


def make_list_of_new_direcs():
    """
    Opens file of already processed files, checks emccd files for new solutions,
    makes list of new directories to process.
    """
    old_list = pd.read_csv("data/emccd_directory_names.csv")
    old_list = old_list["direc_names"].tolist()
    rootdir = "/Volumes/meteor/emccd/pylig/trajectory"
    full_list = [x[0] for x in os.walk(rootdir)][1:]
    new_list = [x for x in full_list if "failures" not in x and x not in old_list]
    return new_list


def extract_from_report_file(fname):
    """
    Pull relevant data from report.txt file for a single file
    """
    with open(fname, "r") as report_data:
        lines = report_data.readlines()
        ind_v = 0
        ind_state = 0
        ind_cov = 0
        for line in lines:
            if ind_v != 0 and ind_state != 0 and ind_cov != 0:
                break
            if ind_v == 0:
                if line.find("Radiant (geocentric") != -1:
                    ind_v = lines.index(line)
            if ind_state == 0:
                if line.find("State vector (ECI, epoch of date)") != -1:
                    ind_state = lines.index(line)
            if ind_cov == 0:
                if line.find("State vector covariance matrix") != -1:
                    ind_cov = lines.index(line)

    vel_zen = " ".join(lines[ind_v : ind_v + 8]).strip().split()
    init_vel = float(vel_zen[vel_zen.index("Vinf") + 2]) * 1000 * 100  # cm/s
    state_vec = lines[ind_state + 1 : ind_state + 7]
    vec_array = []
    for line in state_vec:
        pos = line.strip().split()
        vec_array.append(float(pos[2]))
    cov_matrix = lines[ind_cov + 1 : ind_cov + 7]
    cov_array = []
    for line in cov_matrix:
        row = line.strip().split()
        float_row = [float(x[:-1]) for x in row]
        cov_array.append(float_row)
    return init_vel, state_vec, cov_matrix


def extract_from_table():
    """
    Pull relevant data from solution_table json file
    """
    df_sol_tbl = pd.read_json(
        os.path.join("solution_table.json"),
        orient="records",
    )
    df_filtered = df_sol_tbl[
        [
            "solution_id",
            "solver",
            "utc_ref",
            "begin_height",
            "mass",
            "a",
            "i",
            "q",
            "e",
            "asc_node",
            "omega",
        ]
    ]
    df_filtered["fname"] = (
        df_filtered["solution_id"] + "." + df_filtered["solver"] + ".report.txt"
    )
    df_filtered = df_filtered[
        [
            "fname",
            "utc_ref",
            "begin_height",
            "mass",
            "a",
            "i",
            "q",
            "e",
            "asc_node",
            "omega",
        ]
    ]
    df_filtered.columns = [
        "solution_id",
        "date",
        "begin_height",
        "mass",
        "a",
        "i",
        "q",
        "e",
        "ascending_node",
        "arg_of_peri",
    ]
    return df_filtered


def extract_simfit_json(fname):
    """Pull the initial mass for camo data from the sim_fit.json file"""
    with open(fname) as data_file:
        data = json.load(data_file)
        mass = data["m_init"]
    return mass


def make_full_df(list_of_direcs):
    """
    Takes a list of directories and extracts from the each report.txt file in the directory,
    the meteoroid state vector (x,y,z,vx,vy,vz) in ECI frame (m, m/s), the covariance matrix
    for said state vector, the UTC date and the photometric mass (kg). Then runs a bunch of functions
    to create calculated columns for the dataset like kc parameter.
    """
    solution_ids = []
    init_vels = []
    state_vecs = []
    cov_matrices = []
    failed_files = []
    for i in range(len(list_of_direcs)):
        list_of_sol_ids = [x[2] for x in os.walk(list_of_direcs[i])][0]
        list_of_sol_ids = [y for y in list_of_sol_ids if "pylig.report.txt" in y] # we're not using any of the gural files as they don't have the covariance matrix
        for solution_id in list_of_sol_ids:
            fname = os.path.join(list_of_direcs[i], solution_id)
            try:
                init_vel, state_vec, cov_matrix = extract_from_report_file(
                    fname, "emccd"
                )
                solution_ids.append(solution_id)
                init_vels.append(init_vel)
                state_vecs.append(state_vec)
                cov_matrices.append(cov_matrix)
            except:
                failed_files.append(fname)
    df_report = pd.DataFrame(
        {
            "solution_id": solution_ids,
            "velocities": init_vels,
            "state_vec": state_vecs,
            "cov_matrix": cov_matrices,
        }
    )
    df_table = extract_from_table()
    df = df_report.merge(df_table, how="left", on="solution_id")
    kcs = calc_kc(df)
    df["kc"] = kcs
    df["rho"] = 3.7e3
    df.loc[(df["kc"] >=87.0) & (df["kc"] < 92.0), "rho"] = 2.0e3
    df.loc[(df["kc"] >=92.0) & (df["kc"] < 96.0), "rho"] = 1.0e3
    df.loc[(df["kc"] >=96.0) & (df["kc"] < 103.0), "rho"] = 0.75e3
    df.loc[df["kc"] >=103, "rho"] = 0.27e3
    df["diameter"] = 2 * np.cbrt(3 * df["mass"] / (4 * np.pi * df["rho"]))
    with open(f"failed_files_emccd.txt", "a") as text_file:
        text_file.write(", ".join(failed_files))
    return df


def get_kb_params(fname, ftype):
    """Pull params needed to calculate the kb from report.txt file"""
    with open(fname, "r") as report_data:
        lines = report_data.readlines()
        ind_llal = 0
        ind_vz = 0
        for line in lines:
            if ind_llal != 0 and ind_vz != 0:
                break
            if ind_llal == 0:
                if line.find("Reference point") != -1:
                    ind_llal = lines.index(line)
            if ind_vz == 0:
                if line.find("Radiant (geocentric") != -1:
                    ind_vz = lines.index(line)

    # get group of lines around position due to inconsistencies between files
    if ftype == "pylig":
        date_line = lines[6].strip().split()
    else: # gural
        date_line = lines[3].strip().split()

    lat_lon_alt = " ".join(lines[ind_llal : ind_llal + 10]).strip().split()
    vel_zen = " ".join(lines[ind_vz : ind_vz + 8]).strip().split()
    # Extract data from lines
    day_of_year = int(date_line[1][8:])
    sec_in_day = (
        float(date_line[2][:2]) * 60 * 60
        + float(date_line[2][3:5]) * 60
        + float(date_line[2][6:])
    )
    geo_lat = float(lat_lon_alt[lat_lon_alt.index("geo") + 2])
    geo_lon = float(lat_lon_alt[lat_lon_alt.index("Lon") + 2])
    alt = float(lat_lon_alt[lat_lon_alt.index("Ht") + 2]) / 1000  # km
    init_vel = float(vel_zen[vel_zen.index("Vinf") + 2]) * 1000 * 100  # cm/s
    zen_rad = float(vel_zen[vel_zen.index("Zc") + 2]) * np.pi / 180  # radians

    return day_of_year, sec_in_day, alt, geo_lat, geo_lon, init_vel, zen_rad


def calc_kb(day_of_year, sec_in_day, alt, geo_lat, geo_lon, init_vel, zen_rad):
    """
    kb = log(air density in g/cm^3) + 2.5log(initial velocity in cm/s) - 0.5log(cos(zenith distance of radiant))
    In wmpl, nrlmsis00 code has a gtd7 function which takes input, output and flags classes and returns air density.
    The initial velocity and zenith distance can be pulled from the data directly.
    """
    # Gets the air density
    output_array = nrlmsise_output()
    input_set = nrlmsise_input()
    flags = nrlmsise_flags()
    input_set.doy = day_of_year
    input_set.sec = sec_in_day
    input_set.alt = alt  # in km
    input_set.g_lat = geo_lat
    input_set.g_long = geo_lon
    lstCalc(input_set)
    input_set.f107A = 150.0
    input_set.f107 = 150.0
    input_set.ap = 4.0
    flags.switches[0] = 0
    for i in range(1, 24):
        flags.switches[i] = 1
    try:
        gtd7(input_set, flags, output_array)
        rho = output_array.d[5]  # total mass density in g/cm^3
        kb = (
            np.log10(rho)
            + (2.5 * np.log10(init_vel))
            - (0.5 * np.log10(np.cos(zen_rad)))
        )
        return kb
    except:
        return None



def calc_kc(df):
    """
    Calculates kc value from equation given in Jenniskens 2015:
    kc = begin_height (km) + (2.86-2.00log(v_inf (km/s)))/0.0612
    """
    begin_height = np.array(df["begin_height"].tolist())
    v_inf = np.array(df["velocities"].tolist())
    v_inf = v_inf / 100000
    kc = begin_height + (2.86 - 2.00 * np.log10(v_inf)) / 0.0612
    return kc