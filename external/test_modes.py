import os
import sys
import numpy as np
from typing import Union, List

class Modes:
    def __init__ (self, ansys_working_dir, out_freq_filename="out_freq.txt", out_modes_filename="out_modes.txt"):
        self.ansys_working_dir = ansys_working_dir
        # self.legacy = legacy
        self.out_freq_filename = out_freq_filename
        self.out_modes_filename = out_modes_filename

        self.num_modes = len(self.read_frequencies())


    def read_frequencies(self, path=None):
        """
        Por padrão, lerá as frequências armazenadas no arquivo 'out_freq.txt' dentro de 'working_dir'
        ou entrar caminho completo/relativo para o arquivo
        """
        file = path or os.path.join(self.ansys_working_dir, self.out_freq_filename)
        frequencies = np.loadtxt(file)

        self.num_modes = frequencies.shape[0]

        return frequencies

    def read_modes(self, path: Union[str, List[str]] = None):
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
                num_nodes = int(len(data) / self.num_modes)
                modes_dir = np.reshape(data, (self.num_modes, num_nodes))
                modes_concat.append(modes_dir)
            # concatena nas colunas → resultado [num_modes, num_nodes*ndirs]
            return np.hstack(modes_concat)
        else:
            # caso paths seja string única
            data = np.loadtxt(path)
            num_nodes = int(len(data) / self.num_modes)
            modes = np.reshape(data, (self.num_modes, num_nodes))
            return modes

    def print_modes(self, modes=None):
        if not modes:
            modes = self.read_modes()
        np.set_printoptions(threshold=sys.maxsize)
        print(modes)

path = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - otimização de hiperparâmetros\ANSYS"
out_freq_filename = "target_freq.txt"
out_modes_filename = "target_modes.txt"

# path = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 2\input"
# out_freq_filename = "out_freq.txt"
# out_modes_filename = "out_modos_y.txt"

test = Modes(path, out_freq_filename, out_modes_filename)
test.print_modes()

print("Fim")