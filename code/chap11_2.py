import json
## 서울시 동 경계 지도 데이터 준비하기
geo_seoul = json.load(open('data/bigfile/EMD_Seoul.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo_seoul['features'][0]['properties']

# 위도, 경도 좌표 출력
geo_seoul['features'][0]['geometry']

## 서울시 동별 외국인 인구 데이터 준비하기
import pandas as pd
foreigner = pd.read_csv('data/bigfile/Foreigner_EMD_Seoul.csv')
foreigner.head()

foreigner.info()

foreigner['code'] = foreigner['code'].astype(str)

## 단계 구분도 만들기
bins = list(foreigner['pop'].quantile([0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
bins

# 배경 지도 만들기
map_seoul = folium.Map(location = [37.56, 127],
                       zoom_start = 12,
                       tiles = 'cartodbpositron')

# 단계구분도 만들기
!pip install folium

import folium
folium.Choropleth(
    geo_data = geo_seoul,
    data = foreigner,
    columns = ('code','pop'),
    key_on = 'feature.properties.ADM_DR_CD',
    fill_color = 'Blues',
    nan_fill_color = 'White',
    fill_opacity = 1,
    line_opacity = 0.5,
    bins = bins) \
        .add_to(map_seoul)
        
# 지도 출력
map_seoul.save('map_seoul.html')

