import numpy as np
from scipy.optimize import linear_sum_assignment # usado para pareamento

from typing import Tuple, Optional

class SpecialFun:
    @staticmethod
    def modal_assurance_criterion(mode1, mode2):
        num = np.dot(mode1, mode2) ** 2
        denom = np.dot(mode1, mode1) * np.dot(mode2, mode2)
        return num / denom

    @staticmethod
    def mac_error(base_modes, comp_modes):
        num_modes = base_modes.shape[0]
        # parcelas de erro do MAC
        mac_error = [abs(1 - SpecialFun.modal_assurance_criterion(base_modes[i, :], comp_modes[i, :]))
                      for i in range(num_modes)]
        # soma
        return sum(mac_error)

    @staticmethod
    def norm_freq_errors(base_freq, comp_freq):
        num_modes = len(base_freq)
        # parcelas de erro normalizado das frequências
        freq_errors = [abs((base_freq[i] - comp_freq[i]) / base_freq[i]) for i in range(num_modes)]
        # soma
        return sum(freq_errors)

    @staticmethod
    def mac_matrix(base_modes, comp_modes):
        """
        Calcula a matriz MAC entre mode-sets. Número de modos de referência (Nb) pode ser diferente dos modos numéricos (Nc), desde que possuam o mesmo grau de liberdade (Ndof).
        base_modes: shape (Nb, Ndof)
        comp_modes: shape (Nc, Ndof)
        Retorna mac: shape (Nb, Nc)
        """
        base_modes = np.asarray(base_modes, dtype=float)
        comp_modes = np.asarray(comp_modes, dtype=float)

        # verificações
        if base_modes.ndim != 2 or comp_modes.ndim != 2:
            raise ValueError("base_modes e comp_modes devem ser arrays 2D (modos x dofs).")
        if base_modes.shape[1] != comp_modes.shape[1]:
            raise ValueError(f"Incompatibilidade de DOFs: base={base_modes.shape[1]} "
                             f"!= comp={comp_modes.shape[1]}")

        numerador = np.abs(base_modes @ comp_modes.T)**2 # [modos, modos]
        norma_base = np.sum(base_modes**2, axis=1, keepdims=True) # [modos, 1]
        norma_comp = np.sum(comp_modes**2, axis=1, keepdims=True).T # [1, modos]
        denominador = norma_base @ norma_comp

        # evita divisão por zero
        small = 1e-30
        denominador[denominador < small] = small

        mac = numerador / denominador

        return mac

    @staticmethod
    def pair_modes_mac(comp_freq, comp_modes, base_modes): # matrizes de modos no formato [modos, nós]
        """
        Ordena os modos numéricos com base no pareamento com modos de referência
        """
        mac = SpecialFun.mac_matrix(base_modes, comp_modes)
        # índices para maximizar o MAC:
        base_index, comp_index = linear_sum_assignment(-mac)
        # reordenação dos modos e frequências numéricos:
        paired_comp_modes = comp_modes[comp_index, :]
        paired_comp_freq = comp_freq[comp_index]
        mac_paired = mac[base_index, comp_index]

        return [paired_comp_freq, paired_comp_modes, (1 - mac_paired).sum()]