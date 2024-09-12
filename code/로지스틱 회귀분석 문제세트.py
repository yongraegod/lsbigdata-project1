# 종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미

# 독립변수:
# 골수의 세포성 (CELL)
# 골수편의 백혈구 비율 (SMEAR)
# 골수의 백혈병 세포 침투 비율 (INFIL)
# 골수 백혈병 세포의 라벨링 인덱스 (LI)
# 말초혈액의 백혈병 세포 수 (BLAST)
# 치료 시작 전 최고 체온 (TEMP)

#========================================
# 문제 1. 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
import pandas as pd

# 워킹 디렉토리 설정
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 데이터 로드
df = pd.read_csv("./data/leukemia_remission.txt", sep='\t')
print(df)

## 로지스틱 회귀모델 적합
import statsmodels.api as sm
df['REMISS'] = df['REMISS'].astype('category')


df['REMISS'] = df['REMISS'].replace({'Yes': 1, 'No': 0})


model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()


print(model.summary())

#========================================

# 문제 2. 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.

# 모델 전체 유의성: LLR p-value가 0.04670로, 귀무가설을 기각할 수 있으며, 모델이 통계적으로 유의하다고 볼 수 있다.
# 그러나 개별 변수들 중에서는 LI 변수만 0.1 수준에서 약간의 유의미성을 보이며, 나머지 변수들은 유의수준 0.05에서 유의하지 않다.

#========================================

# 문제 3. 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?

# 2개
# LI: 0.101
# TEMP: 0.198

#========================================

# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?

# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9

# odds = exp(64.2581 + 30.8301 * x1 +  24.6863 * x2 -24.9745 * x3 + 4.3605 * x4 -0.0115 * x5 -100.1734 * x6)

import numpy as np
my_odds = np.exp(64.2581 + 30.8301 * 0.65 +  24.6863 * 0.45 -24.9745 * 0.55 + 4.3605 * 1.2 -0.0115 * 1.1 -100.1734 * 1.0)
# my_odds: 0.03817459641135519

#========================================

# 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?

p_hat = my_odds / (my_odds + 1)
# p_hat: 0.03677088280074742

#========================================

# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.

# TEMP 변수의 계수
temp_coef = -100.1734

# TEMP 변수의 오즈비
temp_odds_ratio = np.exp(temp_coef)
print(f"TEMP 변수의 오즈비: {temp_odds_ratio}")

# TEMP 높을수록 백혈병 세포가 관측되지 않을 확률이 매우 낮아짐

#========================================

# 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.

# CELL 변수의 계수와 표준 오차
cell_coef = 30.8301
cell_std_err = 52.135

# 99% 신뢰구간의 z-값 (양쪽 검정 기준)
from scipy.stats import norm
z_value = norm.ppf(0.995)

# 신뢰구간 계산
lower_bound = cell_coef - z_value * cell_std_err
upper_bound = cell_coef + z_value * cell_std_err

# 오즈비 계산
cell_odds_ratio_lower = np.exp(lower_bound)
cell_odds_ratio_upper = np.exp(upper_bound)

# 결과 출력
print(f"CELL 변수의 99% 오즈비에 대한 신뢰구간:")
print(f"Lower bound (99% CI): {cell_odds_ratio_lower}")
print(f"Upper bound (99% CI): {cell_odds_ratio_upper}")

#========================================

# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 예측 확률 구하기
pred_probs = model.predict(df)

# 50% 기준으로 예측값을 이진 값으로 변환
predicted_classes = np.where(pred_probs >= 0.5, 1, 0)

# 실제값 (종속변수)
actual_classes = df['REMISS']

# 혼동 행렬 구하기
conf_matrix = confusion_matrix(actual_classes, predicted_classes)

print("혼동 행렬:")
print(conf_matrix)

#========================================

# 문제 9. 해당 모델의 Accuracy는 얼마인가요?

# Accuracy 계산
accuracy = accuracy_score(actual_classes, predicted_classes)
print(f"모델의 Accuracy: {accuracy}")
# 0.7407407407407407

#========================================

# 문제 10. 해당 모델의 F1 Score를 구하세요.
f1 = f1_score(actual_classes, predicted_classes)
print(f"모델의 F1 Score: {f1}")
# 0.5882352941176471
