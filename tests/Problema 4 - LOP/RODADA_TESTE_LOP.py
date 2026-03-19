from RODADA_TESTE_main import avaliar_rodada
from tests.indexador_2026 import *

import os

Problema = 4
Compiuter = "DESKTOP"

pb = indexar_problema(Problema)
pc = indexar_device(Compiuter)

base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
local_dir = os.path.join(pc.local_path, f"Problema {Problema}")

script_name = pb.script_filename
noise = pb.noise
parameters = pb.parameters

kv, kh, h_concreto = [1.1e8, 9.7e7, 0.0417]

Econc = 28e9
Emad = 14.5e9

test_params = [Econc, Emad, 210e9, kv, kh, h_concreto, Emad/16, Emad/16, Emad/16]

# Emad = 15e9
#
# test_params = [32.209e9, 15e9, 210e9, 1.1e8, 9.7e7, 0.0417, Emad/16, Emad/16, Emad/16]

avaliar_rodada(parameters, test_params, base_dir, local_dir, script_name, noise)