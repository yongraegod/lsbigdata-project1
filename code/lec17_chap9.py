# p.225 chap.09 한국복지패널 데이터
!pip install pyreadstat

import pandas as pd
import numpy as np
import seaborn as sns

# 데이터 불러오기
raw_welfare = pd.read_spss('data/koweps/Koweps_hpwc14_2019_beta2.sav')

# 복사본 만들기
welfare = raw_welfare.copy()
welfare.shape
welfare.info()
welfare.describe()
