import numpy as np
import random
import os
from pathlib import Path


def gen_noise(n, folder):

    noise = n
    base_path = Path.cwd() / folder / "input"
    f_file = base_path/ f"out_base_freq_{folder}.txt"
    m_file = base_path/ f"out_base_modes_{folder}.txt"

    # Generates noised frequencies file
    frequencies = np.loadtxt(f_file)
    noise_freq = [round(freq + (random.uniform(-noise, noise) * freq),8)  for freq in frequencies] # add noise to the frequency
    (base_path/ f"out_base_freq_{folder}{n}.txt").write_text('\n'.join(map(str, noise_freq))) # save noised frequency on a new folder

    # Generates noised modes file
    modes = np.loadtxt(m_file)
    noise_modes = [round(mode + (random.uniform(-noise, +noise) * mode),8) for mode in modes] # add noise to the modes
    (base_path / f"out_base_modes_{folder}{n}.txt").write_text('\n'.join(map(str, noise_modes))) # save noised modes on a new folder

    return 0