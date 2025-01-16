import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 1 to use

import pandas as pd

data = pd.read_csv('gpascore.csv') #데이터분석 다루는 라이브러리 pandas / 같은 폴더에 든 파일이면 경로 지정 없이 이름만 ㄱㄱ

# 빈 항목 확인 및 처리하는법
# print(data.isnull().sum())
# data = data.dropna() #빈칸있는 부분 제거
# data = data.fillna(100) #빈칸을 100이란 값으로 채움
# print(data.isnull().sum())
# print(data)

# print(data['gpa'])
# print(data['gpa'].max())

# exit()

data = data.dropna()

y데이터 = data['admit'].values #해당 열의 값들을 리스트로 담아줌

x데이터 = []

for i, rows in data.iterrows(): #pandas로 출력한 dataframe 한 행씩 출력-여기선 i에 행 번호 출력될 것
    x데이터.append([ rows['gre'], rows['gpa'], rows['rank'] ])

# print(x데이터)

# exit()


#딥러닝 모델 만드는 법임!

import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='tanh', input_shape=(3,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
 
#결과물 정수로 만들고 싶으면 출력 실수값을 반올림하든..

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#optimizer : 경사하강법에서 기울기 빼는 값 조정
#손실함수 : 확률문제나 분류문제처럼 결과물이 0~1 사이인 경우 binary_crossentropy 쓴다

model.fit( np.array(x데이터), np.array(y데이터), epochs=5000 ) 
#x는 트레이닝 데이터, y는 결과물(0 or 1) 데이터, epochs는 반복학습 횟수
#fit에 list를 바로 집어넣을 수 없어서 numpy 등으로 자료변환 해서 넣어야함

#예측
예측값 = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(예측값)
