import torch
from torchvision import transforms
from torchvision.transforms import v2
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

def get_transforms():
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop((384, 384), scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(15),
        v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandAugment(),
        v2.ColorJitter(0.2, 0.2, 0.2),
        v2.RandomAutocontrast(),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        v2.RandomErasing(p=0.25),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((384, 384), antialias=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_transform, val_transform

# def get_transforms():
#     transform_train = transforms.Compose([
#         transforms.Resize((421, 421)),
#         transforms.RandomCrop(384, padding=8),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
    
#     transform_test = transforms.Compose([
#         transforms.Resize((421, 421)),
#         transforms.CenterCrop(384),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     return transform_train, transform_test