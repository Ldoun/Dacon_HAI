import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor, SiglipForImageClassification

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join('../HAI/', self.df.iloc[index]['img_path'][2:]))
        t_image = self.transform(image)
        return t_image

config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["k_proj", "q_proj"],
        bias="none",
        modules_to_save=["classifier"],
    )
model = SiglipForImageClassification.from_pretrained('google/siglip2-giant-opt-patch16-384', num_labels=393)
model = get_peft_model(model, config)

state = torch.load('?_1/best_model.pt')
model.load_state_dict(state)
model.eval()
model.cuda()

transform_test = transforms.Compose([
        transforms.Resize((421, 421)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

valid_dataset = ImageFolder(root='../HAI/valid', transform=transform_test)
valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=4, #pin_memory=True
)

test_data = pd.read_csv('HAI/test.csv')
test_data.tail()
    
dataset = TestDataset(test_data, transform_test)
loader = DataLoader(
    dataset, batch_size=32, shuffle=False, num_workers=4, #pin_memory=True
)

model.cuda()
with torch.no_grad():
    result = []
    labels = []
    for x in tqdm(loader):
        x = x.cuda()
        output = model(x).logits.softmax(dim=-1).detach().cpu().numpy()
        result.append(output)

idx_to_class = {v: k for k, v in valid_dataset.class_to_idx.items()}

concated = np.concatenate(result)

submission = pd.read_csv('../HAI/sample_submission.csv')

for i in range(concated.shape[1]):
    submission[idx_to_class[i]] = concated[:, i]

submission.to_csv('0.00005_4_final.csv', index=False)
