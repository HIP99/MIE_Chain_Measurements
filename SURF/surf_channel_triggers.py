import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_triggers import SURFChannelTriggers
from SURF_Measurements.surf_channel_info import SURFChannelInfo
from MIE_Chain_Measurements.SURF.surf_channel_mie import SURFChannelMIE

import random
from typing import Any, List

class SURFChannelTriggersMIE(SURFChannelTriggers):
    """
    Automatically finds the MIE RF chain data and stores all the runs

    Inherits from SURFChannelMultiple due to shared methods, and essentially being the same thing. Only this automatically handles MIE data. Do not call parent constructor
    """
    def __init__(self, length:int=1, info:SURFChannelInfo|dict = None, *args, **kwargs):

        self.get_info(info, **kwargs)

        current_dir = Path(__file__).resolve()
        parent_dir = current_dir.parents[2]
        basepath = parent_dir / 'data' / 'SURF_Data' / f'SURF{self.info.surf_channel_name}'
        filename = f'SURF{self.info.surf_channel_name}'

        super().__init__(basepath = basepath, filename = filename, length=length, info=self.info, *args, **kwargs)



def random_surf():
    surf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M']
    pol_names = ['V', 'H']
    return random.choice(surf_names) + random.choice(pol_names) + str(random.randint(1, 8))

if __name__ == '__main__':

    surf_name = random_surf()

    info = {"surf_channel_name":surf_name}

    channel = SURFChannelTriggersMIE(info=info, length=10)

    fig, ax = plt.subplots()
    channel.plot_beamform_samples(ax=ax)

    plt.legend()
    plt.show()