# 교재 p.301 ch.11 지도 시각화

## 시군구 경계 지도 데이터 준비하기
import json
geo = json.load(open('data/bigfile/SIG.geojson', encoding = 'UTF-8'))

# 행정 구역 코드 출력
geo['features'][0]['properties']

# 위도, 경도 좌표 출력
geo['features'][0]['geometry']

## 시군구별 인구 데이터 준비하기
import pandas as pd
df_pop = pd.read_csv('data/bigfile/Population_SIG.csv')
df_pop.head()

df_pop.info()

# 코드가 문자로 되어 있어야 지도 만드는데 활용 가능
df_pop['code'] = df_pop['code'].astype(str) 

## 단계 구분도 만들기
# !pip install folium

# 1. 배경 지도 만들기
import folium
folium.Map(location = [35.95, 127.7],
           zoom_start = 8)

map_sig = folium.Map(location = [35.95, 127.7],
                     zoom_start = 8,
                     tiles = 'cartodbpositron')
map_sig

# 2. 단계 구분도 만들기
folium.Choropleth(
    geo_data = geo,
    data = df_pop,
    columns = ('code', 'pop'),
    key_on = 'feature.properties.SIG_CD') \
        .add_to(map_sig)

# 3. 계급 구간 정하기
bins = list(df_pop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

# 4. 디자인 수정하기
# 배경 지도 만들기
map_sig = folium.Map(location = [35.95, 127.7],
                     zoom_start = 8,
                     tiles = 'cartodbpositron')
                     
# 단계 구분도 만들기
folium.Choropleth(
    geo_data=geo,
    data=df_pop,
    columns=('code', 'pop'),
    key_on='feature.properties.SIG_CD',
    fill_color='YlGnBu',
    fill_opacity=1,
    line_opacity=0.5,
    bins=bins
).add_to(map_sig)

map_sig.save('map_sig.html')

------------------------------------------------------------
import json

geo_seoul = json.load(open("./data/bigfile/SIG_Seoul.geojson", encoding="UTF-8"))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0])
geo_seoul["features"][0].keys()

# 숫자가 바뀌면 "구"가 바뀐다!
geo_seoul["features"][0]["properties"]
geo_seoul["features"][0]["geometry"]

# 리스트로 정보 빼오기
coordinate_list = geo_seoul["features"][0]["geometry"]["coordinates"]
len(coordinate_list[0][0])
coordinate_list[0][0]

# 그래프로 출력
import numpy as np
import matplotlib.pyplot as plt
coordinate_array=np.array(coordinate_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]
# plt.scatter(x,y)
# plt.plot(x[::10], y[::10])
plt.plot(x, y)
plt.show()
plt.clf()
--------------------------
# 함수로 만들기
def draw_seoul(num):
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]
    
    plt.plot(x, y)
    plt.rcParams.update({"font.family" : "Malgun Gothic"})
    plt.title(gu_name)
    plt.show()
    plt.clf()
    
    return None
    
draw_seoul(6)
-------------------------------------------------------
## 서울시 전체 지도 그리기 ##
-------------------------------------------------------
# 빈 리스트 생성
data = []

# 각 구의 이름과 좌표를 리스트에 저장
for i in range(25):
    gu_name = geo_seoul["features"][i]["properties"]['SIG_KOR_NM']
    coordinate_list = geo_seoul["features"][i]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:, 0]
    y = coordinate_array[:, 1]
    
# 각 좌표를 data에 개별 행으로 추가
    for j in range(len(x)):
        data.append({'gu_name': gu_name,
                     'x': x[j],
                     'y': y[j]})

# 데이터프레임 생성
df = pd.DataFrame(data)

# 데이터프레임 출력
print(df)
==========================================================
plt.plot(x, y, hue = "gu_name")

## 구 이름 만들기
gu_name = list()

# 방법 1: for 문으로 하기
for i in range(25):
    # gu_name = gu_name + [geo_seoul["features"][i]["properties"]['SIG_KOR_NM']]
    gu_name.append(geo_seoul["features"][i]["properties"]['SIG_KOR_NM'])

# 방법 2: 리스트 컴프리헨션으로 하기
gu_name = [geo_seoul["features"][i]["properties"]['SIG_KOR_NM'] for i in range(len(geo_seoul['features']))]
gu_name

## gu_name, x, y 판다스 데이터 프레임
def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]['SIG_KOR_NM']
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]
    
    return pd.DataFrame({"gu_name": gu_name, "x" : x, "y": y})

make_seouldf(1)

## 이제 df에 구별 데이터를 쌓기
result = pd.DataFrame({})

for i in range(25):
    result = pd.concat([result, make_seouldf(i)], ignore_index=True)

result

# 그래프로 표시
result.plot(kind='scatter',
            x="x", y="y",
            style='o', s=1)
plt.show()
plt.clf()

# 서울 그래프 그리기
import seaborn as sns
sns.scatterplot(data=result,
                x='x', y='y', hue='gu_name', s=3,
                palette = "viridis",
                legend = False)
plt.show()
plt.clf()
-------------------------------------------------------
# 강남구 vs 강남구 아닌 그룹 차이를 두고 싶다,,,
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"]=="강남구", "강남","안강남"))
gangnam_df

sns.scatterplot(data=gangnam_df,
                x='x', y='y', hue='is_gangnam', s=5,
                palette={"안강남":"grey", "강남":"red"},
                legend = True)
plt.show()
plt.clf()
gangnam_df['is_gangnam'].unique()
