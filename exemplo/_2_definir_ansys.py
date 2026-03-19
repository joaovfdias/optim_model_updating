from external.ansys.parser import Ansys
from _1_definir_dados import diretorio_base as base_dir, nome_script as base_script_filename # importa os dados necessários da etapa 1, é possível renomeá-los na importação

import os
import shutil


# 2. CRIAÇÃO DO OBJETO ANSYS
    # responsável por gerenciar as rodadas a cada avaliação
    # é necessário passar os caminhos de execução e os dados de referência

# para execução do ANSYS, podemos criar uma pasta no mesmo diretório base (automático)
# ou, para não gerar lixo em pastas sincronizadas com a nuvem, por exemplo, definir um diretório local
local_dir = None # preencher se preferir

# CAMINHOS
# executável do Ansys [desnecessário alterar caso não use o metodo Legacy (PyAnsys necessário)]
ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

# diretório de trabalho (onde ele roda)
ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS') # caso não passe o local, cria a pasta ANSYS no diretório base
os.makedirs(ansys_working_dir, exist_ok=True) # cria a pasta se não existir

# diretórios de entrada e saída
input_dir = os.path.join(base_dir, 'input')
output_dir = os.path.join(local_dir if local_dir else base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# copia o modelo base inicial (se existir) para a pasta de execução do ANSYS
unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_{os.getpid()}") # cria pastas isoladas para rodadas
os.makedirs(unique_ansys_dir, exist_ok=True)
try:
    shutil.copy(os.path.join(base_dir, "ModBase.db"), unique_ansys_dir)
except FileNotFoundError:
    print("\nAviso: não existe ModBase.db na pasta base. Nenhuma cópia foi feita.")

# nome dos arquivos base: script e dados de referência
base_script_filename = base_script_filename or "script.mac"
base_freq_filename = "target_freq.txt"
base_modes_filename = "target_modes.txt"

# nome dos arquivos de saída do modelo (definido no script)
out_freq_filename = "out_freq.txt"
out_modes_filename = "out_modes.txt"

# criação e configuração do objeto Ansys
ansys = Ansys(ansys_exe_path, unique_ansys_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir, legacy=False)
ansys.set_output_filenames(out_freq_filename, out_modes_filename)
ansys.max_attempts = 6

