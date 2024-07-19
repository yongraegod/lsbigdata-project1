#lec6 행렬

import numpy as np

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.vstack(
    (np.arange(1, 5),
     np.arange(12, 16))
     )
print("행렬:\n", matrix)


# 빈 행렬 만들기
np.zeros(5)
np.zeros([5,4])

# 행렬 채우면서 만들기
np.arange(1, 5).reshape([2,2])
np.arange(1, 7).reshape((2,3))
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1, 7).reshape((2,-1))

# Q. 0에서 99까지 수 중에서 랜덤하게 50개 숫자를 뽑고,
# 5 by 10 행렬 만드세요.

a = np.random.randint(0, 100, 50).reshape(5,-1)


#order 옵션
np.arange(1, 21).reshape((4,5), order = 'c') #행 우선
np.arange(1, 21).reshape((4,5), order = 'f') #열 우선


#행렬 인덱싱
mat_a = np.arange(1, 21).reshape((4,5), order = 'f')
mat_a[2, 3] #15
mat_a[1, 4] #18
mat_a[0:2, 3] #13, 14
mat_a[1:3, 1:4] #6, 10, 14, 7, 11, 15

#행, 열자리가 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3,::2]
mat_a[:,4]

#짝수 행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20,-1))
mat_b
mat_b[1::2,:]

#선택적으로 골라오기
mat_b[[1,4,6,14],]

# 행렬 필터링
x = np.arange(1, 11).reshape((5, 2)) * 2
x
x[[True,True,False,False,True], 0] #결과가 1차원 벡터로 나옴

mat_b[:,1] #자동으로 차원이 줄어들었다, 벡터
mat_b[:,1].reshape((-1,1)) #벡터를 행렬로 전환
mat_b[:,1:2] #차원이 2차원으로 그대로 유지, 행렬
mat_b[:,(1,)] #차원 유지
mat_b[:,[1,]] #차원 유지


#필터링
#mat_b에서 7의 배수를 확인
mat_b[mat_b[:,1] % 7 == 0,2]


#사진은 행렬이다!
import numpy as np
import matplotlib.pyplot as plt

# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1:\n", img1)

# 행렬을 이미지로 표시
plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
plt.clf()

a = np.random.randint(0, 256, 20).reshape(4,-1)
a / 255   #최대값: 1 / 최소값: 0 / 행렬 안의 값이 0과 1사이에 있도록 설정
plt.imshow(a / 255, cmap='ocean', interpolation='nearest')
plt.colorbar()
plt.show()
plt.clf()


#p.17 이미지 다운로드
import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

# 이미지 읽기
!pip install imageio
import imageio
import numpy as np

jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape) #3차원임을 알게 됨
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

len(jelly)
jelly.shape
jelly[:, :, 0]
jelly[:, :, 0].transpose().shape


plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 0]) #R 
plt.imshow(jelly[:, :, 1]) #G
plt.imshow(jelly[:, :, 2]) #B
plt.imshow(jelly[:, :, 3]) #투명도
plt.axis('off') #축 정보 없애기
plt.show()
plt.clf()


# 3차원 배열

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape # (2,2,3) 2장 겹쳐 있다 / (2,3) 짜리가 / (장,행,열)

# 첫 번째 2차원 배열 선택
first_slice = my_array[0, :, :]
print("첫 번째 2차원 배열:\n", first_slice)

# 두 번째 차원의 세 번째 요소를 제외한 배열 선택
filtered_array = my_array[:, :, :-1]
print("세 번째 요소를 제외한 배열:\n", filtered_array)

my_array[:, :, [0, 2]]
my_array[:, 0, :]
my_array[0, 1, [1,2]] #my_array[0, 1, 1:3]


#5장의 (5,4) 3차원 matrix 생렬
mat_x = np.arange(1, 101).reshape(5,5,4)
mat_x = np.arange(1, 101).reshape(-1,5,2)
mat_x.shape

len(mat_x)

my_array2 = np.array([my_array, my_array])
my_array2.shape # (2,2,2,3) 2wkd 


p.22 넘파이 배열 기본 제공 함수들
a = np.array([[1,2,3], [4,5,6]])
a.shape

#ex)
z = np.arange(36).reshape(3, 4, 3)
z.sum(axis = 0)

a.sum()
a.sum(axis=0).shape
a.sum(axis=1)

a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b=np.random.randint(0,100,50).reshape(5,-1)
mat_b

#가장 큰 수는?
mat_b.max()

#행별로 가장 큰수는?
mat_b.max(axis=1)

#열별로 가장 큰수는?
mat_b.max(axis=0)


#누적합
a=np.array([1,3,2,5])
a.cumsum()

#행별 누적합은?
mat_b.cumsum(axis=1)

#누적곱
a.cumprod()
mat_b.cumprod(axis=1) #행별 누적곱

#1차원 배열로 변환
mat_b.reshape(2,5,5)
mat_b.flatten()

#clip
d = np.array([1, 2, 3, 4, 5])
print("클립된 배열:", d.clip(2, 4))
d.clip(2,4)

#배열을 리스트로 변환합니다.
d.tolist()


