a=(1,2,3) # a=1,2,3
a




#shallow copy - '주소 값'을 복사
a=[1,2,3]
a

b= a
b

a[1]=4
a

b
id(a) #a의 id를
id(b) #b id는 a id

#deep copy - '실제 값'을 복사
a=[1,2,3]
a

b=a[:]
b=a.copy()
b

a[1]=4
a
b
id(a) #둘의 id가 다름
id(b)
