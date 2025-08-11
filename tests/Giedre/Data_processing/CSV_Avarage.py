import numpy as np
import pandas as pd
from pathlib import Path


class CSVProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_csv_files(self):
        return list(self.input_path.rglob("*.csv"))

    def get_parameters(self, file_name):
        name = file_name.lower()
        if 'viga' in name:
            return ['modulo', 'poisson', 'dens', 'rigidez1', 'rigidez2']
        elif 'laje' in name:
            return ['modulo', 'poisson', 'dens', 'rigidez1', 'rigidez2', 'rigidez3', 'rigidez4']
        else:
            return []

    def load_data(self):
        file_groups = {
            "VigaN0": [],
            "VigaN5": [],
            "VigaN15": [],
            "VigaMacN0": [],
            "VigaMacN5": [],
            "VigaMacN15": [],
            "LajeN0": [],
            "LajeN5": [],
            "LajeN15": [],
            "LajeMacN0": [],
            "LajeMacN5": [],
            "LajeMacN15": [],
        }

        for file_path in self.get_csv_files():
            parameters = self.get_parameters(file_path.name)
            df = pd.read_csv(file_path, sep=';', low_memory=False)

            cols = ['Iteration', 'Individual', 'Fitness'] + parameters
            selected_cols = [col for col in cols if col in df.columns]
            df = df[selected_cols].dropna()

            name = file_path.name.lower()
            # Identifica o tipo (viga/laje)
            if 'viga' in name.lower():
                model = 'Viga'
            elif 'laje' in name.lower():
                model = 'Laje'
            else:
                model = None

            # Identifica se é Mac
            mac = 'Mac' if 'mac' in name.lower() else ''

            # Identifica o nível de ruído
            if '15' in name:
                noise = 'N15'
            elif '5' in name:
                noise = 'N5'
            else:
                noise = 'N0'

            # Monta o nome do grupo e adiciona ao dicionário, se for um tipo válido
            if model:
                group = f"{model}{mac}{noise}"
                file_groups[group].append(df)


        return file_groups

    def compute_and_save_averages(self):
        groups = self.load_data()

        for group_name, dfs in groups.items():
            if not dfs:
                print(f"⚠️ Nenhum dado encontrado para {group_name}, pulando...")
                continue

            shapes = [df.shape for df in dfs]
            if len(set(shapes)) > 1:
                print(f"❌ Erro: Os arquivos de '{group_name}' têm formas diferentes: {shapes}")
                continue

            arrays = [df.to_numpy(dtype=np.float64) for df in dfs]
            stacked = np.stack(arrays)
            avg_array = np.mean(stacked, axis=0)
            df_avg = pd.DataFrame(avg_array, columns=dfs[0].columns)

            output_file = self.output_path / f"media_celular_{group_name}.csv"
            df_avg.to_csv(output_file, index=False, sep=';')
            print(f"✅ Média célula-a-célula de '{group_name}' salva em: {output_file}")
            print(df_avg)


# Exemplo de uso
if __name__ == "__main__":
    entrada = Path(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data')
    saida = Path(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\Average_Values')
    processor = CSVProcessor(entrada, saida)
    processor.compute_and_save_averages()