import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate
from scipy.interpolate import UnivariateSpline
from RF_Utils.Scope_Data import Scope_Data
from RF_Utils.Pulse import Pulse

class Impulse_Response(Scope_Data):
    """
    An impulse response measurement taken from an oscilloscope
    Handles general impulse response methods
    It assumes the pulse is channel 1 and the response is channel 2
    """
    def __init__(self, filepath, tag, *args, **kwargs):

        super().__init__(filepath=filepath, tag=tag, *args, **kwargs)

        self.pulse = Pulse(time = getattr(self, "time"), waveform=getattr(self, "C1W1").waveform, tag=tag+"_pulse")

        delattr(self, "C1W1")

        self.response = Pulse(time = self.pulse.time, waveform=getattr(self, "C2W1").waveform, tag=tag+"_response")

        delattr(self, "C2W1")

    def __len__(self):
        return len(self.response)
    
    def __iter__(self):
        return iter(self.response)
    
    def __array__(self):
        return self.response
    
    @property
    def group_delay(self):
        dt = self.time[1] - self.time[0]

        corr = correlate(self.response, self.pulse, mode='full')
        lags = np.arange(-len(self.pulse) + 1, len(self.response)) * dt

        i_peak = np.argmax(corr)
        window = 40  # adjust size as needed 5
        x_window = lags[i_peak - window:i_peak + window + 1]
        y_window = corr[i_peak - window:i_peak + window + 1]

        spline = UnivariateSpline(x_window, y_window, k=3, s=0)
        fine_lags = np.linspace(x_window[0], x_window[-1], 1000)
        fine_corr = spline(fine_lags)
        group_delay = fine_lags[np.argmax(fine_corr)]
        return group_delay
    
    @property
    def gain(self):
        xf, fft_mag  = self.mag_spectrum
        mask = (xf >= 300*1e6) & (xf <= 1200*1e6)
        average_linear_gain = np.mean(fft_mag[mask])
        _, idk  = self.mag_spectrum_db
        return self.lin_to_db(average_linear_gain)

    @property
    def frequency_response(self):
        frequency_response = self.response.fft/(self.pulse.fft + 1e-12)
        return frequency_response

    @property
    def mag_spectrum(self):
        N = len(self.response)
        overall_fft = self.frequency_response
        overall_mag = np.abs(overall_fft[:N//2 + 1])
        return self.response.xf, overall_mag

    @property
    def mag_spectrum_db(self):
        _, overall_mag = self.mag_spectrum
        return self.response.xf, self.lin_to_db(overall_mag)

    @property
    def impulse_response(self):
        return Pulse(waveform=np.fft.ifft(self.frequency_response).real, time = self.pulse.time)
        
    def find_pulse_peak(self):
        return self.pulse.pulse_peak
    
    def find_response_peak(self):
        return self.response.pulse_peak

    def plot_impulse_response(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        impulse_response = self.impulse_response
        impulse_response.plot_waveform(ax=ax, scale=1e3, **kwargs)
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Impulse Time Domain")

    def plot_pulse(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        # mask = (self.time <= 20*1e-9)

        self.pulse.plot_waveform(ax=ax, scale=1e3, **kwargs)
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Pulse Time Domain")

    def plot_response(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        # mask = (self.time >= 35*1e6) & (self.time <= 60*1e-9)

        self.response.plot_waveform(ax=ax, scale=1e3, **kwargs)
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Response Time Domain")

    def plot_data(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.pulse.plot_waveform(ax=ax, scale=1e3, **kwargs)
        self.response.plot_waveform(ax=ax, scale=1e3, **kwargs)
        ax.set_ylabel("Voltage (mV)")
        ax.set_title("Pulse and Impulse response")
        ax.legend()

    def plot_pulse_fft(self, ax: plt.Axes=None, f_start=300, f_stop=1200, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.pulse.plot_fft(ax=ax, f_start=f_start, f_stop=f_stop, log = True, **kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")
        # ax.grid(True)

    def plot_reponse_fft(self, ax: plt.Axes=None, f_start=300, f_stop=1200, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.response.plot_fft(ax=ax, f_start=f_start, f_stop=f_stop, log = True, **kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")
        # ax.grid(True)


    def plot_fft2(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, add_ons = False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()


        """I know this is a round about way but it makes it easier to add extras"""
        self.impulse_response.plot_fft(ax=ax, f_start=f_start, f_stop=f_stop, log = True, scale = len(self)/2,**kwargs)
        if add_ons:
            self.pulse.plot_fft(ax=ax, alpha = 0.7, f_start=f_start, f_stop=f_stop, log = True, **kwargs)
            self.response.plot_fft(ax=ax, alpha = 0.7, f_start=f_start, f_stop=f_stop, log = True, **kwargs)
            ax.legend()

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")
        # ax.grid(True)

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, add_ons = False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if log:
            xf, fft_mag  = self.mag_spectrum_db
        else:
            xf, fft_mag  = self.mag_spectrum

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        ax.plot(xf[mask]/1e6, fft_mag[mask], label=self.tag)
        if add_ons:
            self.pulse.plot_fft(ax=ax, alpha = 0.7, f_start=f_start, f_stop=f_stop, log = True, **kwargs)
            self.response.plot_fft(ax=ax, alpha = 0.7, f_start=f_start, f_stop=f_stop, log = True, **kwargs)
            ax.legend()

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")
        # ax.grid(True)

    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, scale=1.0, log = True, window_size=5, gain=0.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf, fft_mag  = self.mag_spectrum

        buffer_left = (window_size - 1) // 2 + (window_size - 1) % 2
        buffer_right = (window_size - 1) // 2

        maskbuffer = (xf >= (f_start-buffer_left*20)*1e6) & (xf <= (f_stop+buffer_right*10)*1e6)

        xf = xf[maskbuffer]
        mag = fft_mag[maskbuffer] * scale
        window = np.ones(window_size) / window_size
        mag_smoothed = np.convolve(mag, window, mode='same')

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)
        if log:
            mag_smoothed = 20 * np.log10(mag_smoothed + 1e-12) + gain

        ax.plot(xf[mask] / 1e6, mag_smoothed[mask], label=f"{self.tag} (smoothed)", **kwargs)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude" + (" (dB)" if log else ""))
        ax.set_title("FFT Magnitude Spectrum")

if __name__=="__main__":

    from pathlib import Path

    current_dir = current_file = Path(__file__).resolve()

    parent_dir = current_dir.parents[2]

    fig, ax = plt.subplots()

    filepath = parent_dir / 'data' / 'Scope_Data' / f'FullChain_017'
    data = Impulse_Response(filepath=filepath, tag = "Channel 017")

    # data.plot_response(ax=ax)
    # data.plot_impulse_response(ax=ax)

    data.plot_fft(ax=ax, log=True, add_ons=True)
    # data.plot_fft_smoothed(ax=ax, log=True, window_size=5)

    # data.plot_pulse_fft(ax=ax, log=True)

    # filepath = current_dir / 'data' / f'FullChain_20dBatn'
    # data = Impulse_Response(filepath=filepath, tag = "20 dB")
    # print(data.group_delay)
    # print(data.gain)
    # data.plot_fft(ax=ax, log=True)

    plt.legend()
    plt.show()