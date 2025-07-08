import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Impulse.Impulse_Response import Impulse_Response
from Impulse.Setup_Impulse import Setup_Impulse
from MIE_Channel import MIE_Channel

class RF_Impulse(Impulse_Response, MIE_Channel):
    """
    Impulse response measurement for the AMPA and MIE setup.
    This automatically adjusts the data given a setup of cables and attenuators
    """
    def __init__(self, channel, setup : Setup_Impulse = None, *args, **kwargs):
            
            current_dir = Path(__file__).resolve()

            parent_dir = current_dir.parents[2]

            filepath = parent_dir / 'data' / 'Scope_Data' / f'FullChain_{channel}'

            super().__init__(filepath=filepath, tag = f"Scope Channel : {channel}")

            self.info = {"Channel" : channel}
            self.get_info(channel)

            if setup is None:
                self.setup = Setup_Impulse()
            else:
                self.setup = setup

    @property
    def group_delay(self):
        return super().group_delay - self.setup.group_delay
    
    @property
    def gain(self):
        return super().gain
    
    @property
    def fft(self):
        overall_fft = super().fft
        result = overall_fft/self.setup.fft
        return result
    
if __name__ == '__main__':

    ampa = RF_Impulse(channel="017")

    fig, ax = plt.subplots()
    # ampa.plot_fft(ax=ax, f_start=300, f_stop=1200, log=True, add_ons=True)
    # ampa.plot_fft_smoothed(ax=ax, f_start=300, f_stop=1200, log=True, window_size=5)

    for i in range(1, 193):
        channel = f"{i:03d}"
        ampa = RF_Impulse(channel=channel)

        ampa.plot_response(ax=ax)
        # ampa.plot_pulse_fft(ax=ax, log=True, f_start=300, f_stop=1200)
    # ax.legend()
    plt.show()