import os
import shutil
from glob import glob


def create_validation_folder(data_path):
    # merging columns
    for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
        files = glob(os.path.join(data_path, 'train', col1, '*'))
        for file in files:
            shutil.copy(file, os.path.join(data_path, 'train', col2))

    for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
        shutil.rmtree(os.path.join(data_path, 'train', col1))
            
    with open('valid_data.txt', 'r') as f:
        valid_data = list(map(lambda x: x.strip(), f.readlines())) 

    for _file in valid_data:
        file = os.path.join(data_path, 'train', _file)
        os.makedirs(os.path.dirname(file.replace('train', 'valid')), exist_ok=True)
        shutil.copy(file, file.replace('train', 'valid'))
        os.remove(file)