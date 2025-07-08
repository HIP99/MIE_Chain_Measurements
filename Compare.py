import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np

from S_Params.RF_SParam import RF_SParam
from Impulse.RF_Impulse import RF_Impulse
from SURF.SURF_Average import SURF_Average

from S_Params.Setup_SParam import Setup_SParam
from Impulse.Setup_Impulse import Setup_Impulse

setup_s21 = Setup_SParam()
setup_impulse = Setup_Impulse()

channel = "112"

fig, ax = plt.subplots()

s21 = RF_SParam(channel=channel, setup=setup_s21)
impulse = RF_Impulse(channel=channel, setup=setup_impulse)

s21.plot_s21_filtered(ax=ax, compare=False, f_start=300, f_stop=1200)
# s21.plot_S21(ax=ax, f_start=300, f_stop=1200)
impulse.plot_fft_smoothed(ax=ax, f_start=300, f_stop=1200, window_size=5)

surf_name = impulse.info['SURF Channel']

surf = SURF_Average(surf = surf_name)

surf.average_over(window=True)

# surf.data.tag += " (Windowed)"

surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2, window_size=5)

# surf = SURF_Average(surf = surf_name)

# surf.average_over(window=False)

# surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, scale = len(surf)/2)


ax.set_title("Frequency Data from VNA, Scope and SURF data")
plt.legend()
plt.show()
