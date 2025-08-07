from pathlib import Path


class Group:
    def __init__(self, main_folder_path, group_by, sub_group=None):
        self.main_folder = Path(main_folder_path)
        self.files = self.get_files()
        self.group_mode = group_by
        self.sub_group = sub_group if sub_group else self.files
        self.group_by = {"noise": self.group_noise, 'mac': self.group_mac, 'model': self.group_model}

    def get_files(self):
        main_path = Path(self.main_folder)
        return list(main_path.rglob('*.csv'))

    def group_noise(self): #guarda o endereco dos grupos
        N15 = []
        N5 = []
        N0 = []
        for file in self.sub_group:
            if '15' in file.name:
                N15.append(file)
            elif '5' in file.name:
                N5.append(file)
            else:
                N0.append(file)
        return N15, N5, N0

    def group_mac(self):  # guarda o endereco dos grupos
        freq_mac = []
        freq = []
        for file in self.sub_group:
            freq_mac.append(file) if 'mac' in file.name.lower() else freq.append(file)
        return freq_mac, freq

    def group_model(self):
        laje = []
        viga = []
        for file in self.sub_group:
            laje.append(file) if 'laje' in file.name.lower() else viga.append(file)
        return laje, viga

    def group(self):
      return  self.group_by[self.group_mode]()


