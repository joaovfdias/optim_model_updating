import os
import glob
import pandas as pd
from datetime import datetime


def compile_convergence_history(algo_name, expected_params, log_dir, file_id=None):
    """
    Lê os arquivos de log individuais de cada repetição e consolida a convergência
    em um único CSV pronto para plotagem.
    """
    # Procura todos os CSVs na pasta do conjunto
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    csv_files = [f for f in csv_files if os.path.basename(f).startswith(algo_name)]

    if not csv_files:
        print(f"\n[AVISO] Nenhum arquivo de log individual encontrado para {algo_name} em {log_dir}")
        return

    print(f"\n[COMPILANDO] Compilando {len(csv_files)} rodadas de {algo_name}")

    all_runs_data = []

    for f in csv_files:
        try:
            # Lê o CSV (engine python lida melhor com detectores de separador)
            df = pd.read_csv(f, sep=None, engine='python')
        except Exception as e:
            continue

        if 'Fitness' not in df.columns:
            continue

        # 1. Identifica a coluna de passos
        step_col = 'Generation' if 'Generation' in df.columns else (
            'Iteration' if 'Iteration' in df.columns else None)

        cols_to_check = ['Fitness']
        if step_col:
            cols_to_check.append(step_col)

        # 2. "Passa a tesoura" (corta) na primeira linha vazia (NaN)
        # Isso isola os dados puros e remove ttodo o resumo de metadados do final
        invalid_rows = df[df[cols_to_check].isna().any(axis=1)]
        if not invalid_rows.empty:
            first_invalid_idx = invalid_rows.index[0]
            df = df.loc[:first_invalid_idx - 1].copy()

        # 3. Força a conversão para numérico
        # (Obrigatório porque os textos no final podem ter transformado a coluna toda em formato 'object')
        df['Fitness'] = pd.to_numeric(df['Fitness'], errors='coerce')
        if step_col:
            df[step_col] = pd.to_numeric(df[step_col], errors='coerce')

        # Remove eventuais resquícios por precaução
        df = df.dropna(subset=cols_to_check)

        if df.empty:
            continue

        # 4. Agora sim, aplica a lógica de cada algoritmo com os dados 100% limpos
        if algo_name == "BO" or algo_name == "BO_skopt":
            # BO avalia ponto a ponto. O melhor é o mínimo cumulativo até a avaliação 'x'.
            best_so_far = []
            current_best_fit = float('inf')
            current_best_row = None
            for _, row in df.iterrows():
                if row['Fitness'] < current_best_fit:
                    current_best_fit = row['Fitness']
                    current_best_row = row.copy()
                best_so_far.append(current_best_row)
            df_best = pd.DataFrame(best_so_far)
            df_best['Step'] = range(1, len(df_best) + 1)

        else:
            # GA e PSO: Agrupa por geração/iteração e pega o menor fitness
            if step_col:
                idx = df.groupby(step_col)['Fitness'].idxmin()
                df_best = df.loc[idx].sort_values(step_col).copy()
                df_best['Step'] = df_best[step_col].values
            else:
                df_best = df.copy()
                df_best['Step'] = range(1, len(df_best) + 1)

        all_runs_data.append(df_best)

    if not all_runs_data:
        return

    # Descobre o número máximo de passos entre todas as rodadas
    max_steps = max([len(d) for d in all_runs_data])
    consolidated = pd.DataFrame({'Iteracao': range(1, max_steps + 1)})

    # Parâmetros que queremos monitorar na convergência
    params_to_track = ['Fitness'] + list(expected_params.keys())

    for p in params_to_track:
        run_cols = []
        for i, df_run in enumerate(all_runs_data):
            if p not in df_run.columns:
                continue
            run_name = f"Run{i + 1}"
            col_name = f"{run_name}_{p}"
            run_cols.append(col_name)

            temp_df = df_run[['Step', p]].rename(columns={'Step': 'Iteracao', p: col_name})
            consolidated = pd.merge(consolidated, temp_df, on='Iteracao', how='left')

            # ffill() mantém o último valor conhecido caso a rodada tenha estagnado/parado antes
            consolidated[col_name] = consolidated[col_name].ffill()

        if run_cols:
            consolidated[f'Media_{p}'] = consolidated[run_cols].mean(axis=1)
            consolidated[f'Desvio_{p}'] = consolidated[run_cols].std(axis=1)

    # Reordena colunas para a Média e Desvio ficarem na frente para facilitar sua vida
    cols = ['Iteracao']
    for p in params_to_track:
        if f'Media_{p}' in consolidated.columns:
            cols.extend([f'Media_{p}', f'Desvio_{p}'])
            cols.extend([c for c in consolidated.columns if
                         c.endswith(f"_{p}") and not c.startswith("Media") and not c.startswith("Desvio")])

    consolidated = consolidated[cols]

    out_path = os.path.join(log_dir, f"Convergencia_{algo_name}_{file_id if file_id else datetime.now().strftime("%Y%m%d_%H%M%S")}.csv")
    consolidated.to_csv(out_path, sep=';', decimal='.', index=False)
    print(f"[RODADAS COMPILADAS] Histórico de convergência consolidado em: {os.path.basename(out_path)}")