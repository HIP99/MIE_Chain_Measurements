import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Measurements.SURF_Data import SURF_Data
from SURF_Measurements.SURF_Channel import SURF_Channel
from MIE_Chain_Measurements.SURF.SURF_Channel_MIE import SURF_Channel_MIE
from MIE_Chain_Measurements.MIE_Channel import MIE_Channel
import random

class SURF_Averaging(SURF_Data):
    """
    This is like SURF_Average but you write the file name. Doesn't assume SURF naming system. 
    Basically the same and propably should be parent class of SURF_Average but hey ho.
    Issue is with SURF_Channel vs SURF_Channel_MIE
    """
    def __init__(self, filepath:str = None, surf:str = None, surf_index:int = None, channel_index:int = None, *args, **kwargs):
        
        current_dir = current_file = Path(__file__).resolve()

        parent_dir = current_dir.parents[2]

        self.base_path = parent_dir / 'data'
        
        self.filepath = filepath
        self.surf = surf
        self.surf_index = surf_index
        self.channel_index = channel_index
        self.data = None

    def __len__(self):
        return len(self.data)

    def average_over(self, length:int = 999, factor : int = None, window = False, pre=25, post=256-25+10):
        first_run = SURF_Channel(filepath = self.base_path / f"{self.filepath}_0.pkl", surf = self.surf, surf_index = self.surf_index, channel_index = self.channel_index, run=0)
        if window:
            first_run.extract_pulse_window(pre=pre, post=post)
        self.data = first_run.data
        self.data.tag = f"SURF Channel : {self.surf}"
        del first_run

        if factor:
            self.data.upsampleFreqDomain(factor=factor)

        for i in range(length):
            try:
                self.cross_correlate(SURF_Channel(filepath = self.base_path / f"{self.filepath}_{i+1}.pkl", surf = self.surf, surf_index = self.surf_index, channel_index = self.channel_index, run=0), window=window, factor=factor, pre=pre, post=post)
            except Exception as e:
                if self.surf:
                    print(f"Error in get_info for Surf : {self.surf}_Run_{str(i+1)}: {e}")
                else:
                    print(f"Error in get_info for Surf : {self.surf_index}_{self.channel_index}_Run_{str(i+1)}: {e}")
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
    from pathlib import Path

    surf_name='AV1'
    pre=25
    post=256-pre+10

    filepath = Path('SURF_Data') / f'SURF{surf_name}' / f'SURF{surf_name}'

    surf = SURF_Averaging(filepath = filepath, surf=surf_name, surf_index = None, channel_index = None)


    surf.average_over(length=999, window=True, pre=pre, post=post)
    fig, ax = plt.subplots()
    surf.plot_fft(ax=ax,f_start=300, f_stop=1200, log=True, scale = len(surf)/2)
    plt.show()

