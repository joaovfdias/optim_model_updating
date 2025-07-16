import numpy as np
import random
import os
from pathlib import Path
import shutil

def gen_noise(path, n):

    noise = n
    base_path = Path(path)
    f_file = base_path/ "out_base_freq.txt"
    m_file = base_path/ "out_base_modes.txt"
    s_file = base_path / "script.txt"



    noise_dir = Path(fr"C:\Users\giedr\Documents\TCC\ANSYS\ponte\noise{noise}")
    noise_dir.mkdir(parents=True, exist_ok=True)  # cria a pasta se não existir
    out_dir = noise_dir / "out" # Create an output directory in the noise folder
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(s_file, (noise_dir / "script.txt")) #copy the script to the noise folder

    frequencies = np.loadtxt(f_file)
    noise_freq = [freq + (random.uniform(-noise, noise) * freq)  for freq in frequencies] # add noise to the frequency
    (noise_dir / "out_base_freq.txt").write_text('\n'.join(map(str, noise_freq))) # save noised frequency on a new folder

    modes = np.loadtxt(m_file)
    noise_modes = [mode + (random.uniform(-noise, +noise) * mode) for mode in modes] # add noise to the modes
    (noise_dir / "out_base_modes.txt").write_text('\n'.join(map(str, noise_modes))) # save noised modes on a new folder

    return noise_dir