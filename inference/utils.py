import os
import argparse
from PIL import Image
from torch.utils.data import DataLoader, Dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='?', type=str)
    parser.add_argument('--model', default='?', type=str)
    parser.add_argument('--output', default='?', type=str)

    args = parser.parse_args()
    return args
