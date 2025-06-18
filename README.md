### 데이터 Split
- train set의 각 클래스의 10% 데이터를 랜덤하게 valid로 선별
- 중복 클래스를 지우기 위해, 겹치는 클래스들의 폴더를 merge(ImageFolder 데이터셋을 사용하기 위함)

### 학습
python main.py --train_path ~/data/HAI/train --valid_path ~/data/HAI/valid --batch_size 16 --lr $learning_rate --r $r --num_workers 12 --model $trial_name
- hyper-parameters
  - learning rate (0.00005, 0.0001)
  - r (2 4 8 16 32 64)
