- ?_1과 ?_4 폴더 모델의 경우 create_submission.py의 모델 path를 변경 후 실행하시면 됩니다.
- 나머지 모델의 경우 tta_submission.py의 모델 path를 변경 후 실행하시면 됩니다.
- 최종 모델은 생성된 모든 csv 파일의 평균입니다.
- data.pickle에는 각 파일에 사용된 idx_to_class dict가 저장되어 있습니다. 결과가 이상하다면 위 파일 사용해 idx_to_class 덮어씌우시길 바랍니다.

### 실행
python create_submission.py --data_path $data_path --model $model_path --output $output_submission_file

example: python create_submission.py --data_path ../../data/HAI/ --model ../../HAI_Final_Models/\?_1/best_model.pt --output 1.csv

python tta_submission.py --data_path $data_path --model $model_path --output $output_submission_file

example python create_submission.py --data_path ../../data/HAI/ --model ../../HAI_Final_Models/0.00005_4_11/best_model --output 2.csv

```python
import numpy as np
import pandas as pd
from glob import glob

files = glob('*.csv')
v = 1/len(files)
sub_df = pd.DataFrame()

dfs = [pd.read_csv(file) for file in files]
sub_df['ID'] = dfs[0]['ID']

for col in dfs[0].columns[1:]:
    sub_df[col] = np.sum([df[col].values * v for df in dfs], axis=0)
    
sub_df.to_csv('submission/merged_tta_all.csv', index=False)
sub_df.tail()
```
