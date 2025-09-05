import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scipy.ndimage import gaussian_filter1d

from Impulse.Impulse_Response import Impulse_Response

class Attenuator_Impulse(Impulse_Response):
    """
    This doesn't really need its own class
    The only purpose is to be an impulse response measurement for an attenuator with specfic file naming convention so one can simply input the level of attenuation and it'll find the file
    Assuming naming convention FullChain_{dB}dBAtn
    """
    def __init__(self, dB = 20, *args, **kwargs):

        current_dir = Path(__file__).resolve()

        parent_dir = current_dir.parents[2]

        filepath = parent_dir / 'data' / 'Scope_Data' / f'FullChain_{dB}dBAtn'

        super().__init__(filepath=filepath, tag = f"Attenuator_{dB}dB")

        self.pulse.mean_centering(mask = self.pulse.time >= 40*1e-9)
        self.response.mean_centering(mask = self.response.time >= 40*1e-9)

    @property
    def frequency_response(self):
        """
        Noisy frequency reponse
        """
        frequency_response = self.response.fft/(self.pulse.fft + 1e-12)
        mag = np.abs(frequency_response)
        phase = np.unwrap(np.angle(frequency_response))

        mag_smooth = gaussian_filter1d(mag, 3)
        phase_smooth = gaussian_filter1d(phase, 3)

        return mag_smooth * np.exp(1j * phase_smooth)

if __name__ == '__main__':
    atn_20 = Attenuator_Impulse(dB=20)
    atn_30 = Attenuator_Impulse(dB=30)

    fig, ax = plt.subplots()

    # atn_ir.plot_data(ax=ax)

    atn_20.plot_fft(ax=ax)
    atn_30.plot_fft(ax=ax)
    # ax.set_title("Scope Impulse Response 20 dB + Cable Gain spectrum")

    plt.show()