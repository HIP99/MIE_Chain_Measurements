import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Measurements.SURF_Channel import SURF_Channel
from RF_Utils.Pulse import Pulse
import random

class SURF_Channel_MIE(SURF_Channel):
    """
    Surf data is extracted for everything single surf channel (224 total)
    SURF channel only needs the name of the surf channel and it will extract that surfs data from the whole
    """
    def __init__(self, surf:str = None, surf_index:int = None, channel_index:int = None, run:int=None, *args, **kwargs):

        current_dir = Path(__file__).resolve()

        parent_dir = current_dir.parents[2]

        filepath = parent_dir / 'data' / 'SURF_Data' / f'SURF{surf}' / f'SURF{surf}_{run}.pkl'

        super().__init__(filepath = filepath, surf=surf, surf_index = None, channel_index = None, run=run, *args, **kwargs)

    def format_data(self):
        all_data = super(SURF_Channel, self).format_data()

        self.get_surf_index()

        self.data = Pulse(waveform=all_data[self.surf_index][self.channel_index], sample_frequency=3e9, tag = f'SURF : {self.surf}_{self.run}')


    def extract_pulse_window(self, pre=20, post=120):
        self.data.pulse_window(pre=pre, post=post)



if __name__ == '__main__':
    surf = "IH8"
    run=0

    idk = SURF_Channel_MIE( surf=surf, run=run)

    fig, ax = plt.subplots()

    idk.plot_data(ax=ax)

    plt.legend()
    plt.show()