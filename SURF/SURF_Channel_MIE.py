import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_info import SURFChannelInfo
from RF_Utils.Pulse import Pulse
import random
from typing import Any

class SURFChannelMIE(SURFChannel):
    """
    SURF MIE chain data has a specifc file naming system. Finds data from SURF name input
    """
    def __init__(self, info:SURFChannelInfo|dict = None, run:int=None, *args, **kwargs):

        self.get_info(info, **kwargs)

        current_dir = Path(__file__).resolve()
        parent_dir = current_dir.parents[2]
        filepath = parent_dir / 'data' / 'SURF_Data' / f'SURF{self.info.surf_channel_name}' / f'SURF{self.info.surf_channel_name}_{run}.pkl'

        surf_data = SURFData(filepath=filepath)
        
        super().__init__(data = surf_data.format_data()[self.info.surf_index][self.info.rfsoc_channel], info=self.info)
        
        del surf_data

if __name__ == '__main__':
    run=0

    surf = SURFChannelMIE(info={"surf_channel_name":"AV6"}, run=run)

    fig, ax = plt.subplots()

    surf.plot_samples(ax=ax)

    plt.legend()
    plt.show()