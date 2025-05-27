import torch
from torchvision.datasets import ImageFolder

class DataSet(ImageFolder):
    def __init__(self, root, processor):
        super().__init__(root)
        self.processor = processor

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.processor(images=self.loader(path), return_tensors="pt")
            # if self.transform is not None:
            #     sample = self.transform(sample)
            # if self.target_transform is not None:
            #     target = self.target_transform(target)
        return sample, target
