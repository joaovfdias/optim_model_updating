from TuRBO_run import *
from tests.indexador_2026 import *

import os

Problema = 4
Compiuter = "LEST 2"

pb = indexar_problema(Problema)
pc = indexar_device(Compiuter)

base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
local_dir = os.path.join(pc.local_path, f"Problema {Problema}")

script_name = pb.script_filename
noise = pb.noise
parameters = pb.parameters

TuRBO_run(irun=1, parameters=parameters, base_dir=base_dir, local_dir=local_dir, base_script_filename= script_name, noise=noise)