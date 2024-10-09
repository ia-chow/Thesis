import numpy as np
import pandas as pd
import sys
import json
import os
sys.path.append('../')
import source.WesternMeteorPyLib.wmpl.MetSim.MetSim as metsim
import source.WesternMeteorPyLib.wmpl.MetSim.FitSim as fitsim
import source.WesternMeteorPyLib.wmpl.MetSim.GUI as gui
import source.WesternMeteorPyLib.wmpl.MetSim.MetSimErosion as erosion


print(gui.FragmentationContainer(gui.MetSimGUI('usg_metsim_files/1999-01-14/usg_input_jan_1999.txt', usg_input=True), 
                           'usg_metsim_files/1999-01-14/metsim_fragmentation.txt'))