import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter1d

from Impulse.Impulse_Response import Impulse_Response

class Cable_Impulse(Impulse_Response):
    """
    The only purpose is to be an impulse response measurement for the cable setup.
    """
    def __init__(self, *args, **kwargs):

        current_dir = Path(__file__).resolve()

        parent_dir = current_dir.parents[2]

        filepath = parent_dir / 'data' / 'Scope_Data' / f'FullChain_Cables'

        super().__init__(filepath=filepath, tag='Cable')

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

    cbl_ir = Cable_Impulse()


    print(cbl_ir.group_delay)
    fig, ax = plt.subplots()

    # cbl_ir.plot_data(ax=ax)

    cbl_ir.plot_fft(ax=ax)
    # ax.set_title("Scope Impulse Response Cable Gain spectrum")

    plt.show()