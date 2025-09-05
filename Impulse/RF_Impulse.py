import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Impulse.Impulse_Response import Impulse_Response
from Impulse.Setup_Impulse import Setup_Impulse
# from MIE_Channel import MIE_Channel
from RF_Utils.MIE_Channel import MIE_Channel
from RF_Utils.Waveform import Waveform
from RF_Utils.Pulse import Pulse
from scipy.signal.windows import tukey

from scipy.ndimage import gaussian_filter1d

class RF_Impulse(Impulse_Response):
    """
    Impulse response measurement for the AMPA and MIE setup.
    This automatically adjusts the data given a setup of cables and attenuators
    """
    def __init__(self, info:MIE_Channel|dict = None, setup : Setup_Impulse = None, *args, **kwargs):
            
        self.info:MIE_Channel

        if isinstance(info, MIE_Channel):
            self.info = info
        elif isinstance(info, dict):
            self.info = MIE_Channel(**info)
        else:
            raise TypeError("Infomation input is not in the correct form.")
        
        current_dir = Path(__file__).resolve()

        parent_dir = current_dir.parents[2]

        filepath = parent_dir / 'data' / 'Scope_Data' / f'FullChain_{self.info.rf_channel:03d}'

        Impulse_Response.__init__(self, filepath=filepath, tag = f"Scope Channel : {self.info.rf_channel}")

        self.pulse.mean_centering(mask = self.pulse.time >= 40*1e-9)
        self.response.mean_centering(mask = self.response.time <= 40*1e-9)


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
    def frequency_response(self):
        frequency_response = super().frequency_response
        result = frequency_response/(self.setup.frequency_response+1e-12)
        return result    
    
    # @property
    # def impulse_response(self):
    #     frequency_response = self.frequency_response
    #     N = len(frequency_response)
    #     freqs = np.fft.fftfreq(N)
    #     mask = np.abs(freqs) < cutoff_ratio
    #     return Hf * mask
    #     return Pulse(waveform=np.fft.ifft(self.frequency_response).real, time = self.pulse.time)
    
    """
    Need some kind of zig-zag filter or smoother
    """
    def plot_impulse_response2(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        smoothed_impulse_response = Pulse(waveform = gaussian_filter1d(self.impulse_response, 2), time=self.pulse.time)
        smoothed_impulse_response.plot_waveform(ax=ax, scale=1e3, **kwargs)
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Impulse Time Domain")


if __name__ == '__main__':

    chain = RF_Impulse(info = {'rf_channel':"017"})
    fig, ax = plt.subplots()

    # chain.plot_pulse(ax=ax)
    # chain.plot_response(ax=ax)
    # chain.response.plot_samples(ax=ax, scale=1000)
    # chain.plot_impulse_response(ax=ax)

    # print(chain.impulse_response.p2p/chain.pulse.p2p)

    # chain.plot_pulse(ax=ax)
    # chain.plot_impulse_response2(ax=ax)
    chain.plot_impulse_response(ax=ax)

    # chain.plot_fft(ax=ax, log=True)

    ax.legend()
    plt.show()