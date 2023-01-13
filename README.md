# MNIST_FCN_DNN

## 본 Repository는 패스트캠퍼스의 딥러닝 유치원 강좌를 참고로 제작되었습니다.

## 1. 문제 정의
본 Repository는 MNIST의 숫자 손글씨 데이터(0~9,28*28 이미지 데이터)를 **FCN 모델(Fully-Connected Layer Model)**에  학습시켜 숫자 손글씨 데이터가 입력으로 들어왔을 때 해당 데이터가 실제 어떤 숫자를 나타내는지를 분류하는 분류 모델을 생성합니다.
### MNIST 데이터 구성
- 학습 데이터 : 60,000개
- 테스트 데이터 : 10,000개
### 학습 데이터 Split
- Train data : Validation data = 8 : 2
- Train data = 48,000개
- Validation data = 12,000개 

## 2. 모델 구성
- linear transformation : nn.Linear
- activation function : nn.LeakyReLU
- Regularization : BatchNormalization or Dropout

입력으로 모델의 계층의수(n_layers)가 들어오면 MNIST 숫자 데이터의 입력의 크기(784)에서 최종 출력계층의 크기(10) 사이 등차 간격으로 은닉 계층의 뉴런의 개수가 정해집니다. -> utils.py


 