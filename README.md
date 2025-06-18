### 데이터 Split
- train set의 각 클래스의 10% 데이터를 랜덤하게 valid로 선별
- 중복 클래스를 지우기 위해, 겹치는 클래스들의 폴더를 merge(ImageFolder 데이터셋을 사용하기 위함)

```python
import os
import shutil
from glob import glob

# making validation
for folder in glob('train/*'):
    files = glob(os.path.join(folder, '*'))
    print(len(files))

    print(int(len(files) * 0.1))

    print(random.sample(files, int(len(files) * 0.1)))

    for file in random.sample(files, int(len(files) * 0.1)):
        os.makedirs(os.path.dirname(file.replace('train', 'test2')), exist_ok=True)
        shutil.copy(file, file.replace('train', 'test2'))
        os.remove(file)

# merging columns
for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
    files = glob(os.path.join('train', col1, '*'))
    for file in files:
        shutil.copy(file, os.path.join('train', col2))

for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
    files = glob(os.path.join('val', col1, '*'))
    for file in files:
        shutil.copy(file, os.path.join('val', col2))

for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
    shutil.rmtree(os.path.join('train', col1))

for col1, col2 in [['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'], ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'], ['718_박스터_2017_2024', '박스터_718_2017_2024']]:
    shutil.rmtree(os.path.join('valid', col1))
```

### 학습
python main.py --train_path ~/data/HAI/train --valid_path ~/data/HAI/valid --batch_size 16 --lr $learning_rate --r $r --num_workers 12 --model $trial_name
- hyper-parameters
  - learning rate (0.00005, 0.0001)
  - r (2 4 8 16 32 64)
