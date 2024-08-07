# 교재 p.301 ch.11 지도 시각화

## 시군구 경계 지도 데이터 준비하기
import json
geo = json.load(open('data/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo['features'][0]['properties']

# 위도, 경도 좌표 출력
geo['features'][0]['geometry']

## 시군구별 인구 데이터 준비하기
import pandas as pd
df_pop = pd.read_csv('Population_SIG.csv')
df_pop.head()
