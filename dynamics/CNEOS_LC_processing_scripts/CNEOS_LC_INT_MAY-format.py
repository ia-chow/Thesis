# IMPORTS
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

############## CONFIG - YOU CAN CHANGE THESE #################
lc_path = "../events/2019_MO/2019.173.212548_MO"  # path to the LC txt file relative to this script, without the .txt extension

FILE_NAME = f"{lc_path}.txt"  # LC txt file
OUTPUT_INT = f"{lc_path}_INT.csv"  # intensity csv
OUTPUT_MAG = f"{lc_path}_MAG.csv"  # bolometric magnitude csv

##############################################################

data_list = []

t_list = []
I_list = []

latitude = None
longitude = None

def format_exponent(ax, axis='y'):
    # found here: https://stackoverflow.com/questions/31517156/adjust-exponent-text-after-setting-scientific-limits-on-matplotlib-axis

    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment='left'
        verticalalignment='bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment='right'
        verticalalignment='top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float64(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' %expo

        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)

        # Add in a text box at the top of the y axis
        ax.text(x_pos, y_pos, offset_text, transform=ax.transAxes,
               horizontalalignment=horizontalalignment,
               verticalalignment=verticalalignment)
    return ax



def caretNotation(n, decimals=2):
    
    if n is None:
        return None

    if decimals is None:
        decimals = 2
    
    A = "{0:.{1:d}e}".format(n, decimals)
    a, b = A.split("e")
    b = int(b)

    return a + r'x$\mathregular{10^{%d}}$' %b



def lcPlotter(time_window):


    data_list = []

    t_list = [] 
    I_list = []

    latitude = None
    longitude = None


    # EXTRACT DATA
    with open(FILE_NAME, mode="r+", encoding="ISO-8859-1") as f:  # need to use ISO-8859-1 encoding since the May format LCs have a different encoding than the ASCII encoding used by the no images LCs!
     lines = f.readlines()
     # strip trailing (but NOT leading) whitespaces, including newlines, and also removes lines consisting of ONLY whitespace
     lines = [line.rstrip() for line in lines if line.rstrip()]
     for line in lines:
        # print(line[12:-2])
        # intensity readings are tuples, so this block reads the intensity data
        if line[-1] == "," or line[-1] == "]":  # the intensity data is formatted so the last character in the line is either a comma or a closing square bracket (for the last line of the file)
            line = line[12:-2]  # data is formatted so there is either the word "Intensity" (for the first line) or 11 spaces before the intensity data tuple
            line = line.split(",")
            data_list.append(line)

         # this block reads the date and time from the line saying "At XX:XX:XX UT on YY YYY ZZZZ..."
        if "At" in line:
            utc_time = line[3:14]
            date_time = line[18:29]

         # these two blocks read the latitude and longitude 
        if "Lat" in line:
            temp_line = line.split(" ")
            for ww, word in enumerate(temp_line):
                if word == "Lat":
                    latitude = temp_line[ww + 1].strip(',\n.').translate({ord(char): None for char in '° '})  # removes the degree symbol and the space from the latitude value
        if "Lon" in line:
            temp_line = line.split(" ")
            for ww, word in enumerate(temp_line):
                if word == "Lon":
                    longitude = temp_line[ww + 1].strip(',\n.').translate({ord(char): None for char in '° '})  # removes the degree symbol and the space from the longitude value
                    
        # this block reads the peak intensity from the header data
        if "watts" in line:
            temp_line = line.split(" ")
            peak_intensity = float(temp_line[5])  # splits the line with "watts" as ['was', 'determined', 'to', 'be', 'approximately', 'PEAK_INTENSITY', 'watts/steradian,', 'and', 'total', 'radiated', 'flash']
            peak_magnitude = 6-2.5*np.log10(peak_intensity)

        # this block reads the total energy from the header data
        if "energy" in line:
            temp_line = line.split(" ")
            total_energy = float(temp_line[1])  # splits the line with "energy" as ['energy', 'TOTAL_ENERGY', 'joules', '(6000K', 'blackbody', 'model).']
            total_yield = 8.2508*(total_energy / 4.185e12) **0.885
    # MAKE POINTS
    for pair in data_list:
        # print(pair[1])
        t = float(pair[0])
        I = float(pair[1])

        t_list.append(t)
        I_list.append(I)


    t_list = np.array(t_list)
    # print(t_list)
    I_list = np.array(I_list)

    # USING BROWN ET AL 1996 
    M_bol = 6 - 2.5*np.log10(I_list)
    # print(M_bol)


    ### MAKE PLOTS

    fig = plt.figure(figsize=[12,7])
    ax1 = plt.subplot(211)
    ax1.plot(t_list, I_list, color='red', linewidth=2.0)
    ax1.set_ylabel("Intensity [W/ster]")
    ax1.grid(visible=True, which='major', color='red', linestyle='-')
    ax1.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.2)
    #ax1.grid('on')

    ax1.spines['right'].set_color((.8,.8,.8))
    ax1.spines['top'].set_color((.8,.8,.8))


    ax2 = plt.subplot(212)
    ax2.plot(t_list, M_bol, linewidth=7.0)
    ax2.set_xlabel("Time [seconds after {:}]".format(utc_time))
    ax2.set_ylabel("Bolometric Magnitude")
    ax2.grid(visible=True, which='major', color='red', linestyle='-')
    ax2.grid(visible=True, which='minor', color='black', linestyle='-', alpha=0.2)
    #ax2.grid('on')

    ax2.spines['right'].set_color((.8,.8,.8))
    ax2.spines['top'].set_color((.8,.8,.8))

    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])

    xlab = ax1.xaxis.get_label()
    ylab = ax1.yaxis.get_label()

    xlab.set_style('italic')
    xlab.set_size(16)
    ylab.set_style('italic')
    ylab.set_size(16)

    xlab = ax2.xaxis.get_label()
    ylab = ax2.yaxis.get_label()

    xlab.set_style('italic')
    xlab.set_size(16)
    ylab.set_style('italic')
    ylab.set_size(16)

    ax1.xaxis.set_ticks(np.arange(min(t_list), max(t_list)+1, 0.1), minor=True)
    ax2.xaxis.set_ticks(np.arange(min(t_list), max(t_list)+1, 0.1), minor=True)


    # TIME WINDOW PARSE
    area_under_curve = None
    if not (None in time_window):
        beg, end = time_window

        # find beg and end indicies
        beg_indx = min(range(len(t_list)), key=lambda i: abs(t_list[i] - beg))
        end_indx = min(range(len(t_list)), key=lambda i: abs(t_list[i] - end))

        def checkBounds(indx, bounds):

            if indx < bounds[0]:
                indx = bounds[0]

            if indx > bounds[1]:
                indx = bounds[1]

            return indx

        # check if outside of region
        bounds = [0, len(t_list) - 2]
        beg_indx = checkBounds(beg_indx, bounds)
        end_indx = checkBounds(end_indx, bounds)

        index_under_curve = np.arange(beg_indx, end_indx + 1, 1)

        ### Simple box Integration
        area_under_curve = 0
        for ii in index_under_curve:

            dI = (I_list[ii+1] + I_list[ii])/2
            
            dt = t_list[ii+1] - t_list[ii]
            
            dE = dI*dt*4*np.pi

            area_under_curve += dE

        # Plot area on intensity plot
        ax1.fill_between(t_list[index_under_curve], I_list[index_under_curve], color="r", alpha=0.3)


 #    if area_under_curve is None:
 #        ax1.set_title("Fireball: {:} \n Location: {:},  {:} \n  Peak_text_intensity = {:} W/ster       Peak_text_Energy = {:} J \n Peak Intensity = {:} W/ster (Peak Magnitude = {:.1f}) \n Total Radiated Energy = {:} J      Yield = {:.2f} kT"\
 #                    .format(date_time, latitude, longitude, peak_intensity, total_energy,caretNotation(peak_intensity, 2), peak_magnitude, caretNotation(total_energy, 2), total_yield))
 #    else:
 # #       ax1.set_title("Date: {:}          Location: {:}, {:} \n Peak_text_intensity = {:} W/ster          Text_Energy = {:} J \n  Peak Intensity = {:} W/ster      Total_Energy = {:} J \n Yield = {:.2f} kT                 Area_Energy = {:} J \n Peak Magnitude = {:.1f}"\
 # #                   .format(date_time, latitude, longitude, peak_intensity, total_energy,caretNotation(peak_intensity, 2), caretNotation(total_energy, 2), total_yield, caretNotation(area_under_curve, 2), peak_magnitude))       
 #         ax1.set_title("Date: {:}             Location: {:}, {:} \n Peak Intensity = {:} W/ster    Total Optical Energy = {:} J \n  Yield = {:.2f} kT                 Peak Magnitude = {:.1f}"\
 #                    .format(date_time, latitude, longitude, caretNotation(peak_intensity, 2), caretNotation(total_energy, 2), total_yield, peak_magnitude))       
    ax1.set_xlim(min(t_list), max(t_list))
    ax2.set_xlim(min(t_list), max(t_list))

    ######Change limits for MAGNITUDE HERE

    ax2.set_ylim([np.nanmin(M_bol) - 1, -15])  # change upper ylim to scale with the maximum bolometric magnitude (disregarding nans), lower ylim is -15 since noise threshold is -17

    format_exponent(ax1, axis='y')


    plt.gca().invert_yaxis()
    # EDITED TO SAVE FIGURE
    plt.savefig(lc_path + '.png')

    plt.show()

    # OUTPUT CSV FILES
    with open(OUTPUT_INT, "w+") as f:
        for tt in range(len(t_list)):
            f.write("{:}, {:}\n".format(t_list[tt], I_list[tt]))

    with open(OUTPUT_MAG, "w+") as f:
        for tt in range(len(t_list)):
            f.write("{:}, {:}\n".format(t_list[tt], M_bol[tt]))


if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""
~~~~~CNEOS PLOTTER~~~~~
Plots CNEOS light curves
from txt files.
    """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-beg', type=float, help='Start time relative to the reference datetime to find area underneath.')
    arg_parser.add_argument('-end', type=float, help='End time relative to the reference datetime to find area underneath.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    time_window = [cml_args.beg, cml_args.end]

    lcPlotter(time_window)