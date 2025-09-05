import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from Impulse.Impulse_Response import Impulse_Response
from Impulse.Attenuator_Impulse import Attenuator_Impulse
from Impulse.Cable_Impulse import Cable_Impulse
from RF_Utils.Pulse import Pulse

class Setup_Impulse(Impulse_Response):
    """
    This is where we define the auxiliary setup used for impulse response measurements
    This is used in the RF impulse class to adjust readings for how the raw setup performs
    """
    def __init__(self, *args, **kwargs):

        self.atn20 = Attenuator_Impulse(dB=20)
        self.atn30 = Attenuator_Impulse(dB=30)
        self.cable = Cable_Impulse()

        # print(self.atn20.scope_info['RecordLength'])
        self.tag = 'Setup'

        pulses = [self.atn20.pulse, self.atn30.pulse, self.cable.pulse]
        waveforms = np.stack([p.waveform for p in pulses], axis=0)
        avg_waveform = np.mean(waveforms, axis=0)
        self.pulse = Pulse(waveform=avg_waveform, time=pulses[0].time, tag=self.tag+"_pulse")

        conv_waveform = np.convolve(self.pulse.waveform, self.impulse_response.waveform, mode='full')

        dt = self.pulse.time[1] - self.pulse.time[0]
        t0 = self.pulse.time[0] + self.impulse_response.time[0]
        t_conv = np.arange(len(conv_waveform)) * dt + t0

        self.response = Pulse(waveform=conv_waveform, time=t_conv, tag=self.tag + "_response")

    @property
    def group_delay(self):
        return self.atn20.group_delay + self.atn30.group_delay - self.cable.group_delay
    
    @property
    def gain(self):
        return self.atn20.gain + self.atn30.gain - self.cable.gain
    
    @property
    def frequency_response(self):
        H_atn20 = self.atn20.frequency_response
        H_atn30 = self.atn30.frequency_response
        H_cable = self.cable.frequency_response

        frequency_response = H_atn20 * H_atn30 / H_cable
        return frequency_response
    
    @property
    def mag_spectrum(self):
        N = len(self.atn20.response)
        frequency_response = self.frequency_response
        overall_mag = np.abs(frequency_response[:N//2 + 1])
        return self.atn20.response.xf, overall_mag

    @property
    def mag_spectrum_db(self):
        _, overall_mag = self.mag_spectrum
        return self.atn20.response.xf, self.lin_to_db(overall_mag)

if __name__ == '__main__':
    setup = Setup_Impulse()

    fig, ax = plt.subplots()

    setup.plot_fft(ax=ax)
    # setup.plot_fft_smoothed(ax=ax, f_start=300, f_stop=1200, log=True, window_size=5)
    plt.legend()
    plt.show()
