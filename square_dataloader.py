from torch.utils.data import Dataset

class LineOrientationDataset(Dataset):
    def __init__(self, data_file):
        self.samples = []
        self.data_file = data_file;
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, bitmap = self.samples[index]
        return label, bitmap

    def _init_dataset(self):
        with open(self.data_file, 'r') as f:
            data_lines = f.readlines()

        for line in data_lines:
            data = line.split()

            label = data[0]

            bitmap = [int(data[x]) for x in range(1, 5)]

            self.samples.append(label, bitmap)



if __name__ == '__main__':
    import string

    data_root = '/home/syafiq/Data/tes-names/'
    charset = string.ascii_letters + "-' "
    dataset = TESNamesDataset(data_root, charset)
    print(len(dataset))
    print(dataset[420])
