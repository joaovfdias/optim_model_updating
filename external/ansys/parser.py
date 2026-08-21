import subprocess
import os
import numpy as np
import psutil
from datetime import datetime
import time
from typing import Union, List

from .kill_process import kill_ansys_process

from ansys.mapdl import reader as pymapdl_reader

class Ansys:
    """
    ANSYS: entrar com:

    ansys_exe_path: caminho do executável
    working_dir: caminho da pasta onde serão feitas as rodadas e salvos os arquivos de saída do ANSYS (por padrão é a pasta ANSYS no diretório atual)
    base_scripth_path: caminho do script base de comandos do ANSYS (por padrão é o arquivo "script.txt" dentro de 'working_dir')

    ! usar a estrutura r'caminho' para declarar os diretórios de forma apropriada

    num_nodes define a quantidade de pontos usados para descrever os deslocamentos em cada modo de vibração (será retirado, pode ser ignorado)
    """
    def __init__(self, ansys_exe_path, ansys_working_dir=None, input_dir=None, base_script_filename=None, base_freq_filename=None, base_modes_filename=None, output_dir=None, nos_filename=None, legacy=False):
        """
        :param ansys_exe_path: executável do ANSYS
        :param ansys_working_dir: pasta que o ANSYS roda e gera arquivos de saída. se vazio, será em \\ANSYS dentro do caminho atual
        :param input_dir: pasta onde os arquivos de entrada (script, dados de frequencia e modos) estão. se vazio, será em ansys_working_dir
        :param base_script_filename: nome do script base. se vazio, "script.txt"
        :param base_freq_filename: nome do arquivo de frequencia de referência. se vazio, "out_freq.txt"
        :param base_modes_filename: nome do arquivo de modos de referência. se vazio, "out_modos.txt"
        :param output_dir: diretório em que são salvos os script executáveis de cada indivíduo. se vazio, será em input_dir
        :param legacy: por padrão, usa a biblioteca PyAnsys para gerenciar as rodadas. em caso de incompatibilidade, use False para aplicar processo tradicional. para instalar PyAnsys: python -m pip install pyansys[mapdl-all]
        """

        self.current_dir = os.getcwd() # definindo o diretório atual para estabelecer a pasta padrão 'ANSYS'
        self.anstime = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.ansys_exe_path = ansys_exe_path
        self.ansys_working_dir = ansys_working_dir or os.path.join(self.current_dir, 'ANSYS')
        os.makedirs(self.ansys_working_dir, exist_ok=True)
        self.max_attempts = 5

        self.input_dir = input_dir or self.ansys_working_dir
        self.out_dir = os.path.join(output_dir or self.input_dir, self.anstime)
        os.makedirs(self.out_dir, exist_ok=True)
        self.base_script_path = os.path.join(self.input_dir, base_script_filename or 'script.txt')
        self.index = 0 # usado para numerar os script executáveis

        self.legacy = legacy
        self.base_freq = self.read_frequencies(os.path.join(self.input_dir, base_freq_filename or 'out_base_freq.txt'), force_read=True) # melhorar, dar opção de pegar o caminho
        self.num_base_modes = len(self.base_freq) # define o número de modos com base no número de frequências para ajusta a matriz de dados
        if base_modes_filename is None:
            # default: um arquivo só
            path = os.path.join(self.input_dir, 'out_base_modes.txt')
            self.base_modes = self.read_modes(path,force_read=True)
        elif isinstance(base_modes_filename, (list, tuple)):
            # lista de arquivos (x,y,z)
            paths = [os.path.join(self.input_dir, f) for f in base_modes_filename]
            self.base_modes = self.read_modes(paths,force_read=True)
        else:
            # string única
            path = os.path.join(self.input_dir, base_modes_filename)
            self.base_modes = self.read_modes(path,force_read=True)

        self.out_freq_filename = "out_freq.txt" # alteráveis na chamada das funções read
        self.out_modes_filename = "out_modes.txt"

        if not self.legacy:
            self.kill_ansys_process()
            from ansys.mapdl.core import launch_mapdl
            self.mapdl = launch_mapdl(run_location=self.ansys_working_dir, override=True)


    def set_output_filenames(self, out_freq_filename, out_modes_filename): # pode ser passado direto nas funções read
        self.out_freq_filename = out_freq_filename or self.out_freq_filename
        self.out_modes_filename = out_modes_filename or self.out_modes_filename

    def create_input_file(self, parameters_values, parameters_keys):
        # recebe os valores atuais dos parâmetros (parameters_values) e seus respectivos idenfiticadores (parameters_key) para gerar o script executável (script_exe.txt)

        self.index += 1

        with open(self.base_script_path, 'r', encoding='utf-8') as file:
            content = file.read()

        for i, value in enumerate(parameters_values):
            key = parameters_keys[i]
            content = content.replace(f"%{key}%", str(value))

        #content = content.replace(f"%num_nos%", str(self.num_nodes))
        #content = content.replace(f"%num_modos%", str(self.num_modes))

        new_script_path = os.path.join(self.out_dir, f'script_exe_{self.index}.txt')
        with open(new_script_path, 'w', encoding='utf-8') as file:
            file.write(content)

        new_input = new_script_path if self.legacy else content

        # if self.legacy: # interface antiga, com base em subprocess
        #     self.kill_ansys_process()
        #     # self.remove_temp_files()
        #     new_input = new_script_path
        # else: # nova interface com base no PyAnsys
        #     new_input = content

        return new_input

    def remove_temp_files(self): # legacy
        temp_files = [
            # os.path.join(self.ansys_working_dir, "file.out"),
            os.path.join(self.ansys_working_dir, self.out_freq_filename)]

        if isinstance(self.out_modes_filename, (list, tuple)):
            temp_files.extend(os.path.join(self.ansys_working_dir, op) for op in self.out_modes_filename)
        else:
            temp_files.append(os.path.join(self.ansys_working_dir, self.out_modes_filename))

        for file in temp_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                print(f"Arquivo {file} já não existia na pasta")
                # pass  # já não existe, segue em frente
            except PermissionError:
                print(f"Arquivo em uso, não foi possível remover: {file}")

    def exe_ansys(self, input_file): # legacy
        output_file = os.path.join(self.ansys_working_dir, 'file.out')
        command = f'"{self.ansys_exe_path}" -lch -p ansys -dis INTELMPI -np 1 -dir "{self.ansys_working_dir}" -j modeloc -i "{input_file}" -o "{output_file}" -b -s read'

        result = subprocess.run(command, cwd=self.ansys_working_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # verifica manualmente se o código de saída é 8 e ignora (foram gerados avisos e não erros)
        if result.returncode not in (0, 8):
            print(f"Erro: ANSYS retornou código de saída {result.returncode}.")
            print(f"STDERR: {result.stderr}")

    def read_frequencies(self, path=None, force_read = False):
        """
        Por padrão, lerá as frequências armazenadas no arquivo 'out_freq.txt' dentro de 'working_dir'
        ou entrar caminho completo/relativo para o arquivo
        """
        if self.legacy or force_read:
            file = path or os.path.join(self.ansys_working_dir, self.out_freq_filename)
            frequencies = np.loadtxt(file)


        else:
            rst = os.path.join(self.ansys_working_dir, "file.rst")
            result = pymapdl_reader.read_binary(rst)
            frequencies = np.asarray(result.time_values)

        self.num_modes = frequencies.shape[0]

        return frequencies

    def read_modes(self, path: Union[str, List[str]] = None, force_read = False):

        if self.legacy or force_read:
            """
            Lê modos a partir de um ou mais arquivos (x,y,z).
            Cada arquivo contém deslocamentos [nós × modos].
            Se lista de arquivos for passada, concatena os deslocamentos.
            Retorna array de shape [num_modes, num_dofs].
            """
            if path is None:
                if isinstance(self.out_modes_filename, (list, tuple)):
                    path = [os.path.join(self.ansys_working_dir, op) for op in self.out_modes_filename]
                else:
                    path = os.path.join(self.ansys_working_dir, self.out_modes_filename)

            # Se paths for lista de arquivos (x, y, z)
            if isinstance(path, (list, tuple)):
                modes_concat = []
                for p in path:
                    data = np.loadtxt(p)
                    self.num_nodes = int(len(data) / self.num_modes)
                    modes_dir = np.reshape(data, (self.num_modes, self.num_nodes))
                    modes_concat.append(modes_dir)
                # concatena nas colunas → resultado [num_modes, num_nodes*ndirs]
                return np.hstack(modes_concat)
            else:
                # caso paths seja string única
                data = np.loadtxt(path)
                self.num_nodes = int(len(data) / self.num_modes)
                modes = np.reshape(data, (self.num_modes, self.num_nodes))
                return modes
        else:

            nodes_ref = [100, 102, 102, 207, 303, 303, 4, 35, 35]
            rst = os.path.join(self.ansys_working_dir,"file.rst")
            result = pymapdl_reader.read_binary(rst)
            modes = []

            for i in range(self.num_modes):

                nodes_id,disp = result.nodal_solution(i)
                ref = [np.where(nodes_id == node)[0][0] for node in nodes_ref]

                modes.append(disp[ref,1])

            return np.array(modes)



    @staticmethod
    def kill_ansys_process():
        kill_ansys_process()

    # importada de kill_ansys.py
    # @staticmethod
    # def kill_ansys_process():
    #     for proc in psutil.process_iter(['pid', 'name']):
    #         try:
    #             if proc.info['name'] and 'ANSYS.exe' in proc.info['name']:
    #                 print(f"Encerramento forçado do processo {proc.info['name']} (PID {proc.pid})")
    #                 proc.kill()
    #                 time.sleep(0.2)
    #         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
    #             continue

    def cleanup_lock_file(self): # legacy
        lock_path = os.path.join(self.ansys_working_dir, "modeloc.lock")
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                print("Arquivo .lock removido com sucesso.")
            except Exception as e:
                print(f"Erro ao remover arquivo .lock: {e}")

    def run_ansys(self, input_file, frequencies=True, modes=True):

        def try_ansys():
            self.remove_temp_files()

            self.exe_ansys(input_file) if self.legacy else (self.mapdl.clear(), self.mapdl.input_strings(input_file))  # self.mapdl.input_strings(["\CLEAR" + input_file])

            freq_pass = not frequencies or os.path.exists(os.path.join(self.ansys_working_dir, self.out_freq_filename))

            if isinstance(self.out_modes_filename, (list, tuple)):
                for op in self.out_modes_filename:
                    mode_pass = not modes or os.path.exists(os.path.join(self.ansys_working_dir, op))
                    if not mode_pass: break
            else:
                mode_pass = not modes or os.path.exists(os.path.join(self.ansys_working_dir, self.out_modes_filename))

            return freq_pass and mode_pass

        if try_ansys():
            return

        if self.legacy:
            self.kill_ansys_process()
            self.cleanup_lock_file()
        time.sleep(0.02)

        for attempt in range(2, self.max_attempts + 1): # tenta executar o Ansys novamente caso não encontre os arquivos de saída, por {max_attempts} tentativas
            print(f"Saída ausente, tentativa {attempt} de executar ANSYS")

            if try_ansys():
                return

            if self.legacy:
                self.kill_ansys_process()
                self.cleanup_lock_file()
            time.sleep(0.02)

        raise RuntimeError(f"ANSYS falhou após {self.max_attempts} tentativas.")

        # else:
        #     self.mapdl.clear()
        #     self.mapdl.input_strings(input_file) # implementar maneira de verificar se outputs foram gerados corretamente, como no legacy