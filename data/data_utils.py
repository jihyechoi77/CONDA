from torch.utils.data import Dataset
from PIL import Image


class FilepathDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load and process the data
        image = Image.open(self.filepaths[idx]).convert('RGB')

        if self.transform:
            try:
                image = self.transform(image, return_tensors="pt")
            except:
                image = self.transform(image)


        label = self.labels[idx]

        return image, label
