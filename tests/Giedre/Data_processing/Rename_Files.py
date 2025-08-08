from pathlib import Path

from pathlib import Path

class RenameFiles:
    def __init__(self, folder_path):
        self.folder = Path(folder_path)

    def rename_all(self):
        for file in self.folder.rglob('*.csv'):
            name = file.name.lower()

            # Identificar partes do nome
            is_average = 'media' in name or 'avarage' in name or 'average' in name
            avarage = 'Average' if is_average else ''
            model = 'Bridge' if 'laje' in name else 'Beam'
            analysis = 'Freq+Mac' if 'mac' in name else 'Freq'

            if 'n15' in name or 'n0.15' in name:
                noise = 'N15'
            elif 'n5' in name or 'n0.05' in name:
                noise = 'N5'
            else:
                noise = 'N0'

            # Montar nome-base
            base_name_parts = [avarage, model, analysis, noise]
            base_name = '_'.join(part for part in base_name_parts if part)


            if is_average:
                final_name = base_name + '.csv'
            else:
                trial = 1
                while True:
                    final_name = f'{base_name}_T{trial}.csv'
                    new_path = file.with_name(final_name)
                    if not new_path.exists():
                        break
                    trial += 1


            # Gerar novo caminho e renomear
            final_path = file.with_name(final_name)
            print(f'Rename: {file.name} → {final_path.name}')
            file.rename(final_path)


# Chamada
renomear = RenameFiles(r'C:\Users\giedr\PycharmProjects\GitGeral\tests\Giedre\Data_processing\All_data\Avarage')
renomear.rename_all()