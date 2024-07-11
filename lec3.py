#데이터 타입
x = 15.34
print(x, "는 ", type(x), "형식입니다.", sep='') #sep 입력값 사이를 어떻게 채울 것인가

# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

print(a, type(a))
print(b, type(b))


# 여러 줄 문자열 - " 세개를 넣어주면 됨
ml_str = """This is
a multi-line
string"""
print(ml_str, type(ml_str))


# 문자열 결합
greeting = "안녕" + " " + "파이썬!"
print("결합 된 문자열:", greeting)


# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)


# 리스트 - 문자, 숫자, 리스트 자체도 리스트에 들어 갈 수 있음, 리스트는 되게 유연하다

fruit = ["apple", 'banana', "cherry"]
type(fruit)

numbers = [1, 2, 3, 4, 5]
type(numbers)

mixed_list = [1, "Hello", [1, 2, 3]]
type(mixed_list)

# 튜플 생성 예제

a_ls = [10,20,30,40,50,60,70]
type(a_ls)
a_ls[1] = 25
a_ls[1:4]
a_ls[:4]
a_ls

a_tp = (10, 20, 30, 40, 50, 60, 70) # a = 10, 20, 30 과 동일
type(a_tp)
a_tp[0] #0부터 시작
a_tp[1] = 25 # Tuple이라서 20 -> 25 변경이 안됨, 대신 Tuple은 상대적으로 가볍고 빠름
a_tp[1:] #1번째부터 끝까지
a_tp[:3] #해당 인덱스 미만
a_tp[3:] #해당 인덱스 이상
a_tp[1:3] #해당 인덱스 이상 & 미만


b_tp = (10,)
b_tp
type(b_tp)

b_int = (10)
type(b_int)


#사용자 정의함수

def min_max(numbers):
  return min(numbers), max(numbers)

a=[1,2,3,4,5]
type(a)

result = min_max(a)
result
type(result)

result = min_max([1, 2, 3, 4, 5])
result = min_max((1, 2, 3, 4, 5))
print("Minimum and maximum:", result)


# 딕셔너리형 데이터

person = {
'name': 'John',
'age': 30,
'city': 'New York'
}

james = {
  '이름': "james",
  '나이': (49, 27),
  '국적': ["남수단", "수단"]
}

print("Person:", person)
print("james:", james)

#get()을 사용한 값 빼내오기
person.get('name')
james.get('나이')
james.get('나이')[0]
james.get('나이')[:1]

james_age=james.get('나이')
james_age[1]

james_na=james.get('국적')
james_na[0]

#집합(set)
#집합 생성예제
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복된 'apple'은 제거됨, 순서는 제멋대로 나옴
type(fruits)

# 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)
empty_set

empty_set.add('f')
empty_set.add('apple') # 중복된 'apple'은 제거됨
empty_set.add('apple')
empty_set.add('kiwi')
empty_set.remove('kiwi') #집합에 요소가 없으면 에러가 뜸
empty_set.discard('kiwi') #집합에 요소가 없어도 에러 안뜸

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) #합집합
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

age = 10
is_active = True
is_greater = age > 5 # True 반환
is_equal = (age == 5) # False 반환
print("Is active:", is_active)
print("Is age greater than 5?:", is_greater)
print("Is age equal to 5?:", is_equal)

#조건문
a=3
if (a == 2):
 print("a는 2와 같습니다.")
else:
 print("a는 2와 같지 않습니다.")


# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)


set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)


