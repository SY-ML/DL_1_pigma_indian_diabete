import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 시드 설정
    np.random.seed(1)
    tf.random.set_seed(1)

    # 데이터 불러오기
    df = pd.read_csv('pima-indians-diabetes3.csv')
    y = df['diabetes']
    x = df.drop(columns = ['diabetes'])

    # 표준화 (Standardization) - 평균 0, 편차 1의 분포로 바꾸어주기
    scaler = StandardScaler()
    x_prcd = scaler.fit_transform(x) #표준화된 X

    ### 모델 생성
    input_dim = len(x.columns.tolist())

    model = Sequential()
    model.add(Dense(50, input_dim= input_dim, activation='relu')) # 500개의 노드 생성, input feature가 8개, 활성화 함수는 relu로 사용
    model.add(Dense(10, activation= 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))

    ### 모델 컴파일 및 요약정보 확인
    params = {'optimizer':'adam', 'lr': 0.1,'loss': 'binary_crossentropy', 'metrics':['accuracy'], 'epoch': 100, 'batch_size':10, 'validation_split': 0.3}
    model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=params['metrics'])
    model.summary()


    ### 모델 저장 및 업데이트 셋업
    MODEL_DIR = './models/'
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    model_name = MODEL_DIR+'{epoch:02d}-{val_loss:.4f}.hdf5'

    # 체크포인터
    check_pointer = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=1, save_best_only=True)

    # 얼리스타핑
    early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.01, patience=50, verbose=1)


    # 학습
    result = model.fit(x_prcd, y, epochs=params['epoch'], batch_size=params['batch_size'], validation_split=params['validation_split'], callbacks=[check_pointer, early_stopping])


    # LOG
    acc = result.history['accuracy']
    loss = result.history['loss']
    val_loss = result.history['val_loss']
    val_acc = result.history['val_accuracy']

    ### 시각화
    e = np.arange(len(val_acc))
    plt.plot(e, acc, label = 'accuracy')
    plt.plot(e, loss, label = 'loss')

    plt.plot(e, val_acc, label = 'val_accuracy')
    plt.plot(e, val_loss, label = 'val_loss')

    plt.legend()
    plt.show()