import os
from datetime import datetime
import numpy as np
import time
from scipy import stats
import csv

from GA_run_TEST2 import GA_run
from optimization.parameter import Continuous


struct_params = [
            Continuous(20e9, 35e9, 'modulo_viga_1'),
            Continuous(20e9, 35e9, 'modulo_viga_2'),
            Continuous(20e9, 35e9, 'modulo_centro'),

            Continuous(0.1, 0.40, 'poisson'),
            Continuous(2400, 2600, 'dens'),

            Continuous(50e6, 50e8, 'rigidez1'),
            Continuous(50e6, 50e8, 'rigidez2'),
            Continuous(50e6, 50e8, 'rigidez3'),
            Continuous(50e6, 50e8, 'rigidez4')
        ]

selections = ['tournament', 'roulette_wheel']

BASE_DIR = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros\.LEST2"
LOCAL_DIR = r"C:\Users\Thiago Artur\Documents\Problema 2 (local)"
base_script_filename = "script problema 2 (9 param).mac"

log_dir = os.path.join(BASE_DIR, 'meta-opt', f"rodada_GA_teste_selecao_{datetime.now().strftime("%Y%m%d_%H%M%S")}")
log_title = f"GA_test_selection_{selections[0]}_{selections[1]}"
log_path = os.path.join(log_dir, log_title if log_title.endswith('.csv') else f"{log_title}.csv"
)


class SelectionEvaluator:
    def __init__(self):
        self.global_results = []

        # Defina aqui os pesos de importância para cada métrica na sua pontuação final
        self.W_FIT = 0.6  # 50% de importância para a qualidade da solução
        self.W_CV = 0.2  # 30% de importância para a estabilidade/repetibilidade (baixo Coeficiente de Variação)
        self.W_TIME = 0.2  # 20% de importância para a velocidade de execução

    def _norm_percentile(self, value, all_values):
        """
        Calcula o percentil normalizado do valor em relação ao histórico.
        Como queremos MINIMIZAR fitness, CV e tempo, valores menores devem ter
        scores melhores (menores).
        """
        if not all_values:
            return 0.5
        # Retorna a posição relativa de 'value' dentro de 'all_values' (de 0.0 a 1.0)
        return stats.percentileofscore(all_values, value, kind='mean') / 100.0

    def calculate_score(self, fits, times):
        """
        Sua função de cálculo de score adaptada.
        """
        valid_fits = [f for f in fits if np.isfinite(f) and f < 1e5]
        if not valid_fits:
            return 1.0, None, None, None

        # Aplica log10 para suavizar variações bruscas de fitness
        log_fits = [np.log10(abs(f) + 1e-20) for f in valid_fits]

        avg_log_fit = np.mean(log_fits)
        std_log_fit = np.std(log_fits)

        cv = std_log_fit / (abs(avg_log_fit) + 1e-6)
        cv = np.log10(1.0 + cv)

        avg_time = np.mean(times)

        self.global_results.append({
            'log_fit': avg_log_fit,
            'cv': cv,
            'time': avg_time
        })

        all_log_fits = [x['log_fit'] for x in self.global_results]
        all_cvs = [x['cv'] for x in self.global_results]
        all_times = [x['time'] for x in self.global_results]

        n_fit = self._norm_percentile(avg_log_fit, all_log_fits)
        n_cv = self._norm_percentile(cv, all_cvs)
        n_time = self._norm_percentile(avg_time, all_times)

        score = (
                self.W_FIT * n_fit +
                self.W_CV * n_cv +
                self.W_TIME * n_time
        )

        return score, avg_log_fit, cv, avg_time


def extract_fitness(best_individual):
    """
    Função auxiliar para extrair o fitness do objeto retornado pelo seu GA.
    Adapte essa função conforme a estrutura real do objeto retornado por 'rodada.run()'.
    """
    if hasattr(best_individual, 'fitness'):
        return best_individual.fitness
    elif isinstance(best_individual, dict) and 'fitness' in best_individual:
        return best_individual['fitness']
    elif isinstance(best_individual, tuple) or isinstance(best_individual, list):
        return best_individual[0]  # Chute: fitness costuma ser o primeiro ou último elemento

    # Fallback assumindo que best é apenas um número
    return float(best_individual)


def save_results_to_csv(log_path, results_summary):

    with open(log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        # Cabeçalho com os termos individuais
        writer.writerow(['Metodo de Selecao', 'Score Final', 'Fitness Medio (log_fit)', 'CV', 'Tempo Medio'])

        # Preenchimento das linhas
        for method, data in results_summary.items():
            writer.writerow([
                method,
                f"{data['Score']:.4f}",
                f"{data['Avg_Log_Fit']:.4f}",
                f"{data['CV']:.4f}",
                f"{data['Avg_Time']:.2f}"
            ])


def main():
    num_runs = 5
    evaluator = SelectionEvaluator()
    results_summary = {}

    print(f"=== Iniciando Testes de Seleção ({num_runs} rodadas por método) ===")

    for sel_method in selections:  # ['tournament', 'roulette_wheel']
        print(f"\n>> Testando método de seleção: {sel_method.upper()}")

        method_fits = []
        method_times = []

        for i in range(num_runs):
            print(f"  Rodada {i + 1}/{num_runs}...")
            start_time = time.time()

            # Chama a função principal do seu script GA
            best_result = GA_run(
                irun=i,
                base_dir=BASE_DIR,
                parameters=struct_params,
                selection_method=sel_method,
                local_dir=LOCAL_DIR,
                base_script_filename=base_script_filename
            )

            end_time = time.time()
            elapsed = end_time - start_time

            # Extrai o valor numérico do fitness
            fit_value = extract_fitness(best_result)

            method_fits.append(fit_value)
            method_times.append(elapsed)

            print(f"    Fitness obtido: {fit_value:.4f} | Tempo: {elapsed:.2f}s")

        # Calcula o score deste metodo após as 5 rodadas
        score, avg_log_fit, cv, avg_time = evaluator.calculate_score(method_fits, method_times)

        results_summary[sel_method] = {
            'Score': score,
            'Avg_Log_Fit': avg_log_fit,
            'CV': cv,
            'Avg_Time': avg_time,
            'Raw_Fits': method_fits
        }

        print(
            f"  Resultados para {sel_method}: Score={score:.4f}, Fit Médio (log)={avg_log_fit:.4f}, CV={cv:.4f}, Tempo Médio={avg_time:.2f}s")

    save_results_to_csv(log_path, results_summary)
    print(f"\nResultados salvos com sucesso em: {log_path}.csv")

    # Resumo Final
    print("\n" + "=" * 50)
    print("=== RESUMO FINAL DOS MÉTODOS DE SELEÇÃO ===")
    print("=" * 50)

    best_method = None
    best_score = float('inf')

    for method, data in results_summary.items():
        print(f"Método: {method.ljust(15)} | Score Global: {data['Score']:.4f}")
        # Como as métricas foram normalizadas (menor é melhor), o menor score vence.
        if data['Score'] is not None and data['Score'] < best_score:
            best_score = data['Score']
            best_method = method

    print(f"\n=> O método vencedor com base nos pesos estabelecidos foi: **{best_method.upper()}**!")


if __name__ == "__main__":
    main()