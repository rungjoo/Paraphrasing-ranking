from torch.utils.data import Dataset, DataLoader

class AllDatasetLoader(Dataset):
    def __init__(self, txt_file, sample=1.0):
        self.original_list = []
        self.paraphrase_list = []
        self.label_list = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        use_len = int(len(dataset)*sample)
        dataset = dataset[:use_len]
        
        for line in dataset:
            line = line.strip()
            original, paraphrase, label = line.split('\t')
            self.original_list.append(original)
            self.paraphrase_list.append(paraphrase)
            self.label_list.append(label)
        
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, idx):
        return self.original_list[idx], self.paraphrase_list[idx], self.label_list[idx]