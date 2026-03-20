from external.ansys.parser import Ansys
from utils.special_functions import SpecialFun

from typing import Callable, Any


def fitness_function_ansys(keys: list[str], ansys: 'Ansys', preset: int=1, **kwargs) -> Callable[[list[float]], tuple[float, dict[str, Any]]]:
    """
    Presets de funções objetivo para avaliação dos modelos. Expansível.

    Parameters
    ----------
    keys
        Identificadores dos parâmetros conforme definidos no script MAPDL.
    ansys
        Objeto da classe Ansys que gerencia a comunicação com o solver.
    preset
        Define o tipo de função objetivo a ser construída (padrão é 1).
        * 1: Frequências naturais + modos de vibração com pareamento (MAC).
        * 2: Apenas frequências naturais.
    **kwargs
        Parâmetros adicionais opcionais dependentes do preset escolhido.

        Para o Preset 1:
        * wf (float): Peso associado ao erro de frequência.
        * wm (float): Peso associado ao erro de MAC.

    Returns
    -------
    fitness_function
        A função objetivo configurada e pronta para avaliação.
        Essa função interna recebe uma lista de parâmetros numéricos (`params`)
        e retorna uma tupla contendo o valor calculado de `fitness` (float)
        e um dicionário (`info`) com os dados auxiliares para análise ou logging.

    Raises
    ------
    ValueError
        Caso o preset informado não seja suportado.
    """

    def run_model(params: list, frequencies: bool, modes: bool):
        input_file = ansys.create_input_file(params, keys)
        ansys.run_ansys(input_file, frequencies, modes)

    if preset == 1:
        print(f"\nDefinida função usando dados modais de frequências e modos.")
        peso_freq = kwargs.get('wf', 1)
        peso_mac = kwargs.get('wm', 1)

        def fitness_function(params: list) -> tuple[float, dict[str, Any]]:
            """
            Preset 1: Frequências naturais + modos de vibração.
            Analisa o modelo construído com 'params' e retorna o fitness e dicionário de dados adicionais.
            """

            run_model(params, frequencies=True, modes=True)

            comp_freq = ansys.read_frequencies()
            comp_modes = ansys.read_modes()

            paired_comp_freq, paired_comp_modes, mac_error_sum, macs = (
                SpecialFun.pair_modes_mac(
                    comp_freq, comp_modes, ansys.base_modes
                )
            )

            freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

            fit_freq = peso_freq * freq_error_sum
            fit_mac = peso_mac * mac_error_sum
            fitness = fit_freq + fit_mac

            return fitness, {"Fit. (freq.)": fit_freq, "Fit. (MAC)": fit_mac,
                             "Freq.": paired_comp_freq, "Freq. Error": [abs((bf - nf) / bf) for bf, nf in zip(ansys.base_freq, paired_comp_freq)],
                             "Mode": paired_comp_modes, "MAC": macs}

    elif preset == 2:
        print(f"\nDefinida função usando dados modais de frequências.")

        def fitness_function(params: list) -> tuple[float, dict[str, Any]]:
            """
            Preset 2: Frequências naturais.
            Analisa o modelo construído com 'params' e retorna o fitness e dicionário de dados adicionais.
            """

            run_model(params, frequencies=True, modes=False)

            comp_freq = ansys.read_frequencies()

            freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, comp_freq)

            fitness = freq_error_sum

            return fitness, {"Freq.": comp_freq,
                             "Freq. Error": [abs((bf - nf) / bf) for bf, nf in zip(ansys.base_freq, comp_freq)]}

    else:
        raise ValueError(f"Preset {preset} not available.")

    return fitness_function