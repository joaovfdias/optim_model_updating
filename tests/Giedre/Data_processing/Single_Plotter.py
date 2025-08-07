from importlib.metadata import files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from Group import Group
from pathlib import Path

class Plotter:
    def __init__(self, files_path, graph_style):
        self.files_path = [Path(f) for f in files_path]
        self.graph_style= graph_style
        self.plot_style = {"boxplot": self.boxplot, 'scatter': self.scatter, 'scatter_last_gen': self.scatter_last_iteration, 'line': self.line}

    def get_data(self, file):
        df = pd.read_csv(file, sep=';', low_memory=False)
        df = df.dropna(subset=["Iteration", "Fitness"])
        df["Iteration"] = df["Iteration"].astype(str)  # pode ser string se for categórico
        return df

    def graph_title(self,file):
        noise = file.parent.name  # "noise0"
        file_name = file.stem  # "media_celular_laje"
        model = file_name.split("_")[-1]  # "laje"
        plt.title(f"{self.graph_style} {model} {noise}")

    def boxplot(self):
        for file in self.files_path:
            df = self.get_data(file)

            # Cria o boxplot
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=df, x="Iteration", y="Fitness")

            # Personaliza o gráfico
            self.graph_title(file)
            plt.grid(True)
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Fitness", fontsize=12)

            # Garante que tudo apareça corretamente
            plt.tight_layout()

            # Mostra o gráfico
            plt.show()

    def scatter(self):
        for name, file in enumerate(self.files_path):
            df = self.get_data(file)

            self.graph_title(file)
            sns.scatterplot(data=df, x="Iteration", y="Fitness", s=50)
            plt.grid(True)
            plt.show()

    def scatter_last_iteration(self):
        for file in self.files_path:
            df = self.get_data(file)

            last_iteration = df["Iteration"].max()
            df_last_iteration = df[df["Iteration"] == last_iteration]

            sns.scatterplot(data=df_last_iteration, x='Individual', y="Fitness")

            self.graph_title(file)
            plt.xlabel("Individual")
            plt.xlim(0, df["Individual"].max())
            plt.ylabel("Fitness")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def line(self):
        for file in self.files_path:
            df = self.get_data(file)

            sns.lineplot(data=df, x="Iteration", y="Fitness")

            self.graph_title(file)
            plt.xlabel("Iteration")
            plt.gca().xaxis.set_major_locator(MultipleLocator(5))
            plt.ylabel("Fitness")
            plt.gca().yaxis.set_major_locator(MultipleLocator(5))
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot(self):
        self.plot_style[self.graph_style]()




# Gera os grupos de comparacao
main_folder_path = r"C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Mean_Values\TestFolder"

modes = Group(main_folder_path, 'model').group()

analysis = [(Group(main_folder_path, 'mac', m).group()) for m in modes]

noise = [(Group(main_folder_path, 'noise', m).group()) for m in modes]

#Gera os graficos para cada grupo
styles = ['line', 'boxplot', 'scatter', 'scatter_last_gen']
for style in styles:
    for m in modes:
        Plotter(m,style).plot()

    for a1, a2 in analysis:
        Plotter(a1, style).plot()
        Plotter(a2, style).plot()

    for n1, n2, n3 in noise:
        Plotter(n1, style).plot()
        Plotter(n2, style).plot()
        Plotter(n3, style).plot()



