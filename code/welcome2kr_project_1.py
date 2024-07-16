import pandas as pd
import numpy as np

travel = pd.read_csv('data/travel(2016_2019).csv', encoding = 'euc-kr')
travel

#데이터 탐색
travel.head()
travel.tail()
travel.shape
travel.info()
travel.describe()

# 1. 변수 이름 변경 했는지?
travel2 = travel.copy()
travel2 = travel2.rename(columns = {"통계분류" : "월"})
travel2 = travel2.rename(columns = {"항목" : "place"})
travel2 = travel2.rename(columns = {"2016 년" : "2016yr"})
travel2 = travel2.rename(columns = {"2017 년" : "2017yr"})
travel2 = travel2.rename(columns = {"2018 년" : "2017yr"})
travel2 = travel2.rename(columns = {"2019 년" : "2019yr"})
travel2.head()

# 2. 행들을 필터링 했는지?
travel2.query("월 == '1월'")


# 3. 새로운 변수를 생성했는지?
travel_new['yr_total'] = travel_new['2016yr'] + travel_new['2017yr']

# 4. 그룹 변수 기준으로 요약을 했는지?

# 5. 정렬 했는지?
