import os
import time
import numpy as np

from ansys.mapdl.core import launch_mapdl


malha = [0.5, 0.8, 1, 1.25, 1.5, 2]
key = 'malha'

# caminhos
base_dir = r"preencher"
ansys_working_dir = os.path.join(base_dir, 'ANSYS', 'malha')
input_dir = os.path.join(base_dir, 'input', 'teste malha')
base_script_path = os.path.join(input_dir, 'script base teste malha.txt')
output_dir = os.path.join(input_dir, 'output')

base_freq_path = os.path.join(input_dir, "target_freq.txt")
base_modes_path = os.path.join(input_dir, "target_modes.txt")

mapdl = launch_mapdl(run_location=ansys_working_dir, override=True)

frequencies = []
delta_frequencies = []
elapsed_times = []

for i, m in enumerate(malha):

    with open(base_script_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace(f"%{key}%", str(m))

    new_script_path = os.path.join(output_dir, f'script_base_malha_{m}.txt')
    with open(new_script_path, 'w', encoding='utf-8') as file:
        file.write(content)

    input_file = content

    inicio = time.time()
    mapdl.clear()
    mapdl.input_strings(input_file)
    fim = time.time()
    elapsed_times[i] = fim - inicio

    frequencies[i] = np.loadtxt(base_freq_path)
    delta_frequencies[i] = (frequencies[i] / frequencies[0]) / frequencies[0]

    print(f"\n\n{i+1}.----- Malha de {m} m -----"
          f"\n Tempo: {elapsed_times[i]:.6f} sec")
    print("\n".join(f"f{i}: {f:8.4f} Hz | delta: {df:.4e}"
                    for f, df in zip(frequencies[i], delta_frequencies[i])))

