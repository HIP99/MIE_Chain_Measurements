import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Measurements.SURF_Data import SURF_Data
from MIE_Chain_Measurements.SURF.SURF_Channel_MIE import SURF_Channel_MIE
from MIE_Chain_Measurements.MIE_Channel import MIE_Channel
import random

class SURF_Average(SURF_Data, MIE_Channel):
    """
    Each SURF data aquisition should have 100 pickle files of data. Typically an oscillscope would average over based on a trigger.
    This class takes all the 'triggers' aligns the pulses and takes the superposition
    This method is currerently incredibly geared towards pulses rather than any continuous waveform
    """
    def __init__(self, surf:str = "AV1", *args, **kwargs):
        self.info = {'SURF Channel' : surf}
        self.get_info(surf = surf)
        self.data = None

    def __len__(self):
        return len(self.data)

    def average_over(self, length:int = 999, factor : int = None, window = False, pre=25, post=256-25+10):
        first_run = SURF_Channel_MIE(surf=self.SURF, run=0)
        if window:
            first_run.extract_pulse_window(pre=pre, post=post)
        self.data = first_run.data
        self.data.tag = f"SURF Channel : {self.info['SURF Channel']}"
        del first_run

        if factor:
            self.data.upsampleFreqDomain(factor=factor)

        for i in range(length):
            try:
                self.cross_correlate(SURF_Channel_MIE(surf=self.SURF, run=i+1), window=window, factor=factor, pre=pre, post=post)
            except Exception as e:
                print(f"Error in get_info for Surf : {self.SURF}_Run_{str(i+1)}: {e}")
                break

        ## This is because the triggers peak identification isn't perfect. Fiducial volume of about 4
        self.data.shorten_waveform(5,pre+post-5)

        self.data.waveform /= (length+1)

    def cross_correlate(self, surf_run : SURF_Channel_MIE, factor : int = None, window = True, pre=200, post=400):
        if window:
            surf_run.extract_pulse_window(pre=pre, post=post)
        compare_data = surf_run.data
        if factor:
            compare_data.upsampleFreqDomain(factor=factor)

        corr = np.correlate(self.data - self.data.mean, compare_data - compare_data.mean, mode='full')
        lags = np.arange(-len(compare_data) + 1, len(self.data))
        max_lag = lags[np.argmax(corr)]

        compare_data.correlation_align(self.data, max_lag)

        self.data.waveform += compare_data

        del surf_run
        del compare_data

    def plot_data(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax)
        # ax.set_xlabel('Samples')
        ax.set_ylabel('Raw ADC Counts')

    def plot_data(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax)
        # ax.set_xlabel('Samples')
        ax.set_ylabel('Raw ADC Counts')


    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft(ax=ax, log = log, mask=mask, **kwargs)


    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, window_size=11, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_fft_smoothed(ax=ax, log = log, f_start=f_start, f_stop=f_stop,window_size=window_size, **kwargs)

def random_surf():
    surf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M']
    pol_names = ['V', 'H']
    return random.choice(surf_names) + random.choice(pol_names) + str(random.randint(1, 8))


###We have a potential bad correlation issue messing with the windowing. Need to ensure that it correlates to within the window
###IH8 is an instance of this. Correltation should be either -pre-post or pre+post from zero. Maybe only look in this region

if __name__ == '__main__':

    surf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M']
    pol_names = ['V', 'H']
    # surf_names = ['I', 'J', 'K', 'L', 'M']
    surf_name = surf_names[0]+'V'

    ##
    # pre=25
    # post=256-pre+10
    # fig, ax = plt.subplots()
    # for i in range(2):
    #     surf_name = random_surf()
    #     surf = SURF_Average(surf = surf_name)
    #     surf.average_over(window=True, pre=pre, post=post)
    #     surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)   
    # plt.legend()
    # plt.show()
    ##

    ##This is purely to see if the SURF directories and pickles are there
    # for name in surf_names:
    #     print(name)
    #     for i in range(8):
    #         print(i+1)
    #         surf = SURF_Average(surf=name+"H"+str(i+1))
    #         surf.average_over()

    pre=25
    post=256-pre+10

    surf = SURF_Average(surf="AV1")
    surf.average_over(length=999, window=True, pre=pre, post=post)

    fig, ax = plt.subplots()

    # surf.plot_data(ax=ax)

    surf.plot_fft(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)

    # surf.plot_fft_smoothed(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)

    plt.legend()
    plt.show()