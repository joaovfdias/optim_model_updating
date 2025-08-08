import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Carregar os dados ===
main_file = r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Best-Avarage\Final_Data.csv'
ref_file = r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Best-Avarage\Parameters_Reference.csv'  # <<< Seu arquivo de referência

# Leitura dos arquivos
df = pd.read_csv(main_file, sep=';')
ref_df = pd.read_csv(ref_file, sep=';')

# === 2. Agrupamento e cálculo ===
grouped = df.groupby(['Model', 'Analysis', 'Noise'])
average_df = grouped.mean(numeric_only=True).reset_index()
min_df = grouped.apply(lambda g: g.loc[g['Fitness'].idxmin()]).reset_index(drop=True)

# === 3. Definir parâmetros
common_params = ["Young's Modulus", "Poisson's Ration", "Density", "Stiffness1", "Stiffness2"]
bridge_only_params = ["Stiffness3", "Stiffness4"]

# === 4. Gerar gráficos agrupados por (Model, Analysis)
group_keys = average_df[['Model', 'Analysis']].drop_duplicates()

for _, key in group_keys.iterrows():
    model = key['Model']
    analysis = key['Analysis']

    # Filtrar os dados do grupo atual
    avg_grp = average_df[(average_df['Model'] == model) & (average_df['Analysis'] == analysis)]
    min_grp = min_df[(min_df['Model'] == model) & (min_df['Analysis'] == analysis)]

    # Parâmetros aplicáveis ao modelo
    parameters = common_params.copy()
    if model.lower() == 'bridge':
        parameters += bridge_only_params

    # Iterar por cada parâmetro
    for param in parameters:
        noises = avg_grp['Noise'].astype(str).tolist()
        x = np.arange(len(noises))
        width = 0.35

        avg_vals = avg_grp[param].values
        min_vals = min_grp[param].values

        # Criar gráfico
        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, avg_vals, width, label='Average')
        plt.bar(x + width/2, min_vals, width, label='Minimum (Best)')

        # Buscar e traçar valor de referência
        ref_row = ref_df[(ref_df['Model'] == model) & (ref_df['Parameter'] == param)]
        if not ref_row.empty:
            ref_val = ref_row['Reference'].values[0]
            plt.axhline(y=ref_val, color='red', linestyle='--', label='Reference')

        # Estilização
        plt.title(f'{param} - Model: {model} | Analysis: {analysis}')
        plt.xlabel('Noise')
        plt.ylabel(param)
        plt.xticks(x, noises)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
