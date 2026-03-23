import os
import glob
import pandas as pd

log_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros\.LEST2\meta_opt\populational logs\rodada 1"

# pega todos os CSVs da pasta
csv_files = glob.glob(os.path.join(log_dir, "meta_opt_*.csv"))

if not csv_files:
    raise RuntimeError("Nenhum arquivo CSV encontrado na pasta.")

dfs = []

for f in csv_files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)  # rastreabilidade
        dfs.append(df)
    except Exception as e:
        print(f"[AVISO] Falha ao ler {f}: {e}")

# concatena tudo
df_all = pd.concat(dfs, ignore_index=True)

print(f"{len(df_all)} avaliações carregadas.")

best = df_all.loc[df_all["Score"].idxmin()]

print("\n=== MELHOR RESULTADO GLOBAL ===")
print("Arquivo:", best["source_file"])
print("Score:", best["Score"])
print("Avg_LogFit:", best["Avg_LogFit"])
print("CV:", best["CV"])
print("Avg_Time:", best["Avg_Time"])

print("\nHiperparâmetros ótimos:")
print(best.drop(["Timestamp", "Score", "Avg_LogFit", "CV", "Avg_Time", "source_file"]))