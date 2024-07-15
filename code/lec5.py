import numpy as np

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
#random.seed - 랜덤 값을 고정시킴
np.random.seed(2024)

a = np.random.randint(1, 21, 10)
print(a)

# 두 번째 값 추출
print(a[1])

a[::2] #처음부터 끝까지 두칸씩 건너뛰면서 뽑아
a[0:6:2] #첫번째 값부터 6번째 값까지 두칸씩 뛰면서 뽑아
a[-2] #맨 끝에서 두번째

#Q. 1에서부터 1,000 사이 3의 배수의 합은?

sum(np.arange(3,1001,3))
x = np.arange(3,1001)
sum(x[::3])


print(a)
np.delete(a, [1,3])


a > 3
a[a>3]


np.random.seed(2024)
a = np.random.randint(1, 10000, 300)
a < 5000
a[a < 5000]

#5.1.2 논리 연산자와 조건문
np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a > 2000) & (a < 5000)]

#교재 p.72
import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])
model_names = np.array(df.index)

#연비 15이상 25이하인 차가 몇대?
sum((np_df >= 15) & (np_df <= 25))

#15 이상 20 이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <= 20)]

#평균 mpg보다 높은(이상) 자동차 대수는?
sum(np_df >= np.mean(np_df))

#평균 mpg보다 높은(이상) 자동차 모델은?
model_names[np_df >= np.mean(np_df)]

#평균 mpg보다 낮은(미만) 자동차 모델은?
model_names[np_df < np.mean(np_df)]


#15보다 작거나 22이상인 데이터 개수는?
sum((np_df < 15) | (np_df >= 22))


np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A","B","C","F","W"])
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]

#5.1.4 필터링을 이용한 벡터 변경
a[a > 3000] = 3000
a

#5.1.5 조건을 만족하는 위치 탐색 np.where
np.random.seed(2024)
a = np.random.randint(1, 100, 10)
a < 50
np.where(a < 50) #True가 있는 자리의 index 찾기

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

#처음으로 22000보다 큰 숫자가 나왔을때, 숫자 위치와 그 숫자는?
x = np.where(a > 22000)
type(x)
my_index = x[0][0]
a[my_index]
a[np.where(a > 22000)][0]

#처음으로 10000보다 큰 숫자가 나왔을때,
#50번째로 나오는 숫자 위치와 그 숫자는? 81번째, 21052
y = np.where(a > 10000)
type(y)
my_index = y[0][49]
a[my_index]

#처음으로 500보다 작은 숫자들 중,
#가장 마지막으로 나오는 숫자 위치와 그 숫자는? 960번째, 391
z = np.where(a < 500)
z[0][-1]
my_index = z[0][-1]
a[my_index]


# 5.1.7 빈칸을 나타내는 방법
import numpy as np
a = np.array([20., np.nan, 13., 24., 309.])
a + 3
np.mean(a)
np.nanmean(a)
np.nan_to_num(a, nan = 150)

#5.1.7.b 값이 없음을 나타내는 None
a = None
b = np.nan
a
b
a + 1
b + 1
np.isnan(a)

#5.1.8 빈 칸을 제거하는 방법
np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

#5.1.9 벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

#벡터는 한가지 타입만 받을 수 있으니 str 타입으로 만들어주기
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

#5.1.10 여러 벡터들을 묶어보자
combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec


#5.1.10.a np.column_stack()와 np.row_stack()
col_stacked = np.column_stack((np.arange(1, 5),
                               np.arange(12, 16)))
col_stacked

#Deprecation Warning 더 이상 지원 안하니깐 vstack 써라고 알림
row_stacked = np.row_stack((np.arange(1, 5),
                            np.arange(12, 16)))
row_stacked

# 5.1.10.b 길이가 다른 벡터 합치기
uneven_stacked = np.column_stack((np.arange(1, 5),
                                  np.arange(12, 18)))
uneven_stacked

vec1 = np.arange(1,5)
vec2 = np.arange(12,18)

np.resize(vec1, len(vec2))
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked

#홀수번째 원소
a = np.array([12,21,35,48,5])
a[0::2]

#최대값
a = np.array([1,22,93,64,54])
a.max()

#중복된 값 제거
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

#주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터 생성
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
a
b
#np.array([21,24,31,44,58,67])

#빈 공간 만들기
x = np.empty(6)

#홀수
x[0::2] = a

#짝수
x[[1,3,5]] = b

x
