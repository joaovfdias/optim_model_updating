import os
import time
import numpy as np
import pandas as pd  # Biblioteca essencial para manipular tabelas e CSV
from ansys.mapdl.core import launch_mapdl

malha = [0.25, 0.5, 0.8, 1, 1.25, 1.5, 2]
key = 'malha'

# Caminhos
base_dir = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros\.NOTEBOOK"
ansys_working_dir = os.path.join(base_dir, 'ANSYS', 'malha')
input_dir = os.path.join(base_dir, 'input', 'teste malha')
base_script_path = os.path.join(input_dir, 'script base teste malha.mac')
output_dir = os.path.join(input_dir, 'output')

# Arquivos que o ANSYS gera
base_freq_path = os.path.join(ansys_working_dir, "target_freq.txt")

# Cria diretórios se não existirem
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(ansys_working_dir):
    os.makedirs(ansys_working_dir)

# Inicia o MAPDL
mapdl = launch_mapdl(run_location=ansys_working_dir, override=True)

# Lista para acumular os dados de todas as rodadas
dados_consolidados = []
ref_frequencies = None  # Variável para guardar as frequências da malha 0.5 (referência)

print(f"{'=' * 60}\nINICIANDO TESTE DE SENSIBILIDADE DE MALHA\n{'=' * 60}")

for i, m in enumerate(malha):
    print(f"\n>>> Rodando Malha: {m} ...")

    # 1. Preparação do Script
    with open(base_script_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Substituição da variável de malha
    content = content.replace(f"%{key}%", str(m))

    # Salva script modificado para log/debug
    new_script_path = os.path.join(output_dir, f'script_base_malha_{m}.txt')
    with open(new_script_path, 'w', encoding='utf-8') as file:
        file.write(content)

    # 2. Execução do ANSYS
    start_time = time.time()

    mapdl.clear()  # Limpa a memória do ANSYS para a nova rodada
    mapdl.input_strings(content)  # Envia o script

    end_time = time.time()
    elapsed = end_time - start_time

    # 3. Leitura dos Resultados
    # Verifica se o arquivo foi criado para evitar crash
    if os.path.exists(base_freq_path):
        current_freqs = np.loadtxt(base_freq_path)
        # Garante que seja um array 1D mesmo se houver só 1 frequência
        current_freqs = np.atleast_1d(current_freqs)
    else:
        print(f"[ERRO] Arquivo de frequências não encontrado para malha {m}")
        current_freqs = np.array([])

    # 4. Cálculo do Desvio (Erro Relativo)
    # Assume que a primeira rodada (i=0, malha=0.5) é a referência (erro zero)
    if i == 0:
        ref_frequencies = current_freqs
        deltas = np.zeros_like(current_freqs)
    else:
        # Evita divisão por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            deltas = (current_freqs - ref_frequencies) / ref_frequencies

    # 5. Estruturação dos Dados para o CSV
    # Cria um dicionário para esta linha
    row_data = {
        'Malha': m,
        'Tempo (s)': elapsed
    }

    # Adiciona colunas dinamicamente para cada modo (Freq e Delta)
    for idx, (f, d) in enumerate(zip(current_freqs, deltas)):
        row_data[f'Freq_Modo_{idx + 1}'] = f
        row_data[f'Delta_Modo_{idx + 1}'] = d

    dados_consolidados.append(row_data)

    # 6. Print Formatado no Console
    print(f"Tempo: {elapsed:.4f} s")
    print(f"{'Modo':<5} | {'Freq (Hz)':<12} | {'Desvio (%)':<12}")
    print("-" * 35)
    for idx, (f, d) in enumerate(zip(current_freqs, deltas)):
        print(f"{idx + 1:<5} | {f:<12.4f} | {d * 100:<12.4f}%")

# --- Finalização e Salvamento ---

# Encerra o ANSYS
mapdl.exit()

# Cria o DataFrame e Salva em CSV
df_resultado = pd.DataFrame(dados_consolidados)
csv_path = os.path.join(base_dir, 'resultado_sensibilidade_malha.csv')

# Organiza as colunas (Malha e Tempo primeiro)
cols = ['Malha', 'Tempo (s)'] + [c for c in df_resultado.columns if c not in ['Malha', 'Tempo (s)']]
df_resultado = df_resultado[cols]

df_resultado.to_csv(csv_path, index=False, sep=';', decimal=',')
print(f"\n\n{'=' * 60}\nTESTE CONCLUÍDO. Resultados salvos em:\n{csv_path}\n{'=' * 60}")
print(df_resultado)