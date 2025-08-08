from importlib.metadata import files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from Group import Group
from pathlib import Path



class Multi_Plotter:
    def __init__(self, files_path, graph_style):
        self.files_path = [Path(f) for f in files_path]
        self.graph_style= graph_style
        self.plot_style = {"boxplot": self.boxplot, 'scatter': self.scatter, 'scatter_last_gen': self.scatter_last_iteration, 'line': self.line}

    def get_data(self, file):
        df = pd.read_csv(file, sep=';', low_memory=False)

        # Remove espaços dos nomes das colunas
        df.columns = df.columns.str.strip()

        # Debug: verificar nomes das colunas
        if "Iteration" not in df.columns or "Fitness" not in df.columns:
            print(f"[ERRO] Colunas não encontradas em {file.name}")
            print("Colunas disponíveis:", df.columns.tolist())
            return None  # <-- evita erro adiante

        # Drop de valores nulos
        df = df.dropna(subset=["Iteration", "Fitness"])

        # Ajuste de tipo para Iteration
        if self.graph_style == "boxplot":
            df["Iteration"] = pd.to_numeric(df["Iteration"], errors='coerce')
        else:
            df["Iteration"] = df["Iteration"].astype(str)

        return df

    def graph_title(self,file):
        noise = file.parent.name  # "noise0"
        file_name = file.stem  # "media_celular_laje"
        model = file_name.split("_")[-1]  # "laje"
        plt.title(f"{self.graph_style} {model} {noise}")

    def boxplot(self):
        all_data = []

        for file in self.files_path:
            df = self.get_data(file)
            df["Source"] = Path(file).stem  # identifica o File de origem
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)

        plt.figure(figsize=(16, 6))
        sns.boxplot(data=df_all, x="Iteration", y="Fitness", hue="Source")

        plt.title("Boxplot per Iteration - Average Values")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="File")
        plt.show()

    def scatter(self):
        all_data = []

        for file in self.files_path:
            df = self.get_data(file)
            df["Source"] = Path(file).stem
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)

        plt.figure(figsize=(14, 6))
        sns.scatterplot(data=df_all, x="Iteration", y="Fitness", hue="Source", s=50)

        plt.title("Scatterplot of Fitness per Iteration - Average Values")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="File")
        plt.show()

    def scatter_last_iteration(self):
        all_data = []

        for file in self.files_path:
            df = self.get_data(file)
            last_iteration = df["Iteration"].max()
            df_last = df[df["Iteration"] == last_iteration].copy()
            df_last["Source"] = Path(file).stem
            all_data.append(df_last)

        df_all = pd.concat(all_data, ignore_index=True)

        plt.figure(figsize=(14, 6))
        sns.scatterplot(data=df_all, x="Individual", y="Fitness", hue="Source", s=50)

        plt.title("Scatterplot of Last Generation - Avarage Values")
        plt.xlabel("Individual")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="File")
        plt.show()

    def line(self):
        all_data = []

        for file in self.files_path:
            df = self.get_data(file)
            df["Source"] = Path(file).stem
            all_data.append(df)

        df_all = pd.concat(all_data, ignore_index=True)

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=df_all, x="Iteration", y="Fitness", hue="Source")

        plt.title("Fitness Convergence - Average Values")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(5))
        plt.grid(True)
        plt.tight_layout()
        plt.legend(title="File")
        plt.show()

    def plot(self):
        self.plot_style[self.graph_style]()

# Gera os grupos de comparacao
main_folder_path = r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data\Avarage"

models = Group(main_folder_path, 'model').group()

analysis = [(Group(main_folder_path, 'mac', m).group()) for m in models]

# for i in analysis:
#     noise1 = [(Group(main_folder_path, 'noise', item).group()) for item in i]

noise = [Group(main_folder_path, 'noise', m).group() for m in models]

for line in noise:
    print('line\n')
    for n in line:
        print(f'{n} \n')

#Gera os graficos para cada grupo
styles = ['line', 'boxplot', 'scatter', 'scatter_last_gen']
for style in styles:
#     # for m in models:
#     #     Multi_Plotter(m,style).plot()
#     #
    for a1, a2 in analysis:
        Multi_Plotter(a1, style).plot()
        Multi_Plotter(a2, style).plot()
#     #
#     # for n1, n2, n3 in noise_viga:
#     #     Multi_Plotter(n1, style).plot()
#     #     Multi_Plotter(n2, style).plot()
#     #     Multi_Plotter(n3, style).plot()
#
    for line in noise:
        print('line\n')
        for n in line:
            print(f'{n} \n')
            Multi_Plotter(n, style).plot()
    # for n1, n2, n3 in noise:
    #     print('\n', n1)
    #     Multi_Plotter(n1, style).plot()
    #     print('\n', n2)
    #     Multi_Plotter(n2, style).plot()
    #     print('\n', n3)
#         Multi_Plotter(n3, style).plot()



