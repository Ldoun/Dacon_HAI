import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, SiglipForImageClassification

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from utils import get_args

args = get_args()

class TestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(args.data_path, self.df.iloc[index]['img_path'][2:]))
        t_image = self.transform(image)
        return t_image


if __name__ == "__main__":
    transform_test = transforms.Compose([
            transforms.Resize((421, 421)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        
    
    model = SiglipForImageClassification.from_pretrained(args.model, num_labels=393).cuda()
    test_data = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    test_data.tail()
        
    dataset = TestDataset(test_data, transform_test)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=12, #pin_memory=True
    )
    
    with open('data.pickle', mode='rb') as f:
        idx_to_class = pickle.load(f)

    results = []
    with torch.no_grad():
        for x in tqdm(loader):
            x = x.cuda()

            # TTA: original
            outputs_orig = model(x).logits
            probs_orig = F.softmax(outputs_orig, dim=1)

            # TTA: horizontal flip
            images_flip = torch.flip(x, dims=[3])
            outputs_flip = model(images_flip).logits
            probs_flip = F.softmax(outputs_flip, dim=1)

            # Average
            probs_avg = (probs_orig + probs_flip) * 0.5

            results.append(probs_avg.detach().cpu().numpy())

    concated = np.concatenate(results)
    submission = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'))
    for i in range(concated.shape[1]):
        submission[idx_to_class[i]] = concated[:, i]

    submission.to_csv(args.output, index=False)
