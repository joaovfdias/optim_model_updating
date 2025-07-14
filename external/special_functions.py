import numpy as np


class SpecialFun:
    @staticmethod
    def modal_assurance_criterion(mode1, mode2):
        num = np.dot(mode1, mode2) ** 2
        denom = np.dot(mode1, mode1) * np.dot(mode2, mode2)
        return num / denom

    @staticmethod
    def mac_error(base_modes, comp_modes):
        num_modes = len(base_modes)
        # parcelas de erro do MAC
        mac_error = [abs(1 - SpecialFun.modal_assurance_criterion(base_modes[:, i], comp_modes[:, i]))
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