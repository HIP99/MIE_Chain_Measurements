import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np

from S_Params.RF_SParam import RF_SParam
from Impulse.RF_Impulse import RF_Impulse
# from SURF.SURF_Average import SURF_Average

from S_Params.Setup_SParam import Setup_SParam
from Impulse.Setup_Impulse import Setup_Impulse

from SURF.input_pulse import input_pulse
from scipy.interpolate import interp1d

setup_s21 = Setup_SParam()
setup_impulse = Setup_Impulse()

channel = "112"

fig, ax = plt.subplots()

s21 = RF_SParam(channel=channel, setup=setup_s21)
impulse = RF_Impulse(channel=channel, setup=setup_impulse)

s21.plot_s21_filtered(ax=ax, compare=False, f_start=300, f_stop=1200)
# s21.plot_S21(ax=ax, f_start=300, f_stop=1200)
impulse.plot_fft_smoothed(ax=ax, f_start=300, f_stop=1200, window_size=5)

# surf_name = impulse.info['SURF Channel']

# surf = SURF_Average(surf = surf_name)

# ##
# pulse = input_pulse()
# interp = interp1d(pulse.xf, pulse.pulse_fft, kind='linear', bounds_error=False, fill_value=0)

# pre=25
# post=256-pre+10

# surf = SURF_Average(surf="AV1")
# surf.average_over(length=999, window=True, pre=pre, post=post)
# surf.plot_fft_smoothed(ax=ax, f_start=300, f_stop=1200, log=True, scale = len(surf)/2)

# xf, surf_data = surf.data.get_smoothed_fft(ax=ax,f_start=300, f_stop=1200, scale = len(surf)/2)

# pulse_resampled = interp(xf)


# surf_data = surf_data / pulse_resampled


# mask = (xf >= 300*1e6) & (xf <= 1200*1e6)

# ax.plot(xf[mask]/1e6, 20*np.log10(surf_data[mask]), label = 'Adjusted')
##


# surf.average_over(window=True)

# surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2, window_size=5)


# ax.set_title("Frequency Data from VNA, Scope and SURF data")

#######

# fig, ax1 = plt.subplots()

# x_ref, base = s21.bandwidth_lin()

# xf, surf_data = surf.data.get_smoothed_fft(ax=ax,f_start=300, f_stop=1200, scale = len(surf)/2)

# xf = xf[mask]
# change = surf_data[mask]

# interp = interp1d(xf, change, kind='linear', bounds_error=False, fill_value=0)

# resampled = interp(x_ref)

# filter = resampled/base

# ax1.plot(x_ref/1e6, 20*np.log10(filter))

# ax1.set_xlabel("Frequency (MHz)")
# ax1.set_ylabel("Gain (dB)")
# ax1.set_title("Rough estimate at SURF Low Pass Filter")

#######

ax.legend()
plt.show()
