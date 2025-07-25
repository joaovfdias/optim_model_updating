import numpy as np
from scipy.optimize import linear_sum_assignment # usado para pareamento

from contextlib import contextmanager
import threading
import sys
import time

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
        numerador = np.abs(base_modes @ comp_modes.T)**2 # [modos, modos]
        norma_base = np.sum(base_modes**2, axis=1, keepdims=True) # [modos, 1]
        norma_comp = np.sum(comp_modes**2, axis=1, keepdims=True).T # [1, modos]
        denominador = norma_base @ norma_comp
        # if np.any(np.isclose(denominador, 0)):
        #     raise RuntimeError('Denominator cannot be zero — invalid or ill-conditioned vibration mode detected.')
        return numerador / denominador

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


class Utilities:

    @staticmethod
    @contextmanager
    def display_process(message, status=True):
        if not status:
            yield
            return

        stop = False

        def animate_dots():
            dots = ["", ".", "..", "...", "..", ".", ""]
            while not stop:
                for d in dots:
                    if stop:
                        break
                    sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
                    sys.stdout.write(f"\r{message}{d} ")  # limpa conteúdo da animação
                    sys.stdout.flush()
                    time.sleep(0.5)

        t = threading.Thread(target=animate_dots)
        t.start()

        try:
            yield
        except Exception as e:
            stop = True
            t.join()
            sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
            sys.stdout.flush()
            return e
        finally:
            stop = True
            t.join()
            sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
            sys.stdout.flush()