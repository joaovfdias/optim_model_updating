from TuRBO_run import TuRBO_run
from tests.indexador_2026 import *

from multiprocessing import Process
import os


def run_TuRBO(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, initial_points=None, evaluations=None, batch_size=4, acqf="ts"):
    TuRBO_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise, initial_points, evaluations, batch_size, acqf)

if __name__ == '__main__':

    Problema = 3
    Compiuter = "LEST 2"
    runs = 4

    pb = indexar_problema(Problema)
    pc = indexar_device(Compiuter)

    base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
    local_dir = os.path.join(pc.local_path, f"Problema {Problema}")

    script_name = pb.script_filename
    noise = pb.noise
    parameters = pb.parameters

    for irun in range(1, runs+1):
        print(f"\nRunning TuRBO ({irun}/{runs}). . .")

        p = Process(target=run_TuRBO, args=(irun, parameters, base_dir, local_dir, None, script_name, noise))
        p.start()
        p.join()