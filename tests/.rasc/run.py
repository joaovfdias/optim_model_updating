from tests.BO_ANS_TestRun import BO_run_laje
from multiprocessing import Process


def run(input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir, fitness_metric, acq_func="PI", xi=None, run_idx=1):
    BO_run_laje(input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir, fitness_metric, run_idx, noise_level, xi)

if __name__ == '__main__':

    input_dir = r"D:\Users\Thiago\OneDrive\Documentos\2025.1\Cilamce\Rodadas\inputs"
    base_script_filename = "script_laje.txt"
    output_dir = r"D:\Users\Thiago\OneDrive\Documentos\2025.1\Cilamce\Rodadas\inputs\out"

    initial_points = 70
    evaluations = 490
    acq_func = 'PI'
    xi = 0.01
    fitness_metrics = ['freq+mac', 'freq']
    noise_levels = [0, 0.05, 0.15]
    num_runs = 4
    xis = [0.1, 0.001] # para testar

    for run_idx in range(1, num_runs+1):
        for noise_level in noise_levels:
            for fitness_metric in fitness_metrics:

                # se for preciso pular rodadas desnecessárias / já executadas:
                if noise_level == 0 and fitness_metric == fitness_metrics[0] and run_idx <= 2:
                    continue

                # estrutura do nome dos arquivos de entrada, pode mudar/ser a mesma:
                base_freq_filename = f"out_base_freq_laje_{noise_level}noise_{run_idx}.txt"
                base_modes_filename = f"out_base_modes_laje_{noise_level}noise_{run_idx}.txt"

                p = Process(target=run, args=(input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir, fitness_metric, acq_func, xi, run_idx))
                p.start()
                p.join()