import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable
from skimage import io

# 定数定義
TrainN = 2000  # 学習サンプル総数
TestN = 10000  # テストサンプル総数
ClassN = 10  # クラス数（今回は10）
Size = 28  # 画像サイズ（今回は縦横ともに28）
TrainFile = './Images/TrainingSamples/{0:1d}-{1:04d}.png'
TestFile = './Images/TestSamples/{0:1d}-{1:04d}.png'

# 画像のロード
x_train = np.zeros((TrainN, Size, Size), dtype=np.float32)
t_train = np.zeros(TrainN, dtype=np.int32)
x_test = np.zeros((TestN, Size, Size), dtype=np.float32)
t_test = np.zeros(TestN, dtype=np.int32)

i = 0
for label in range(ClassN):
    for sample in range(TrainN // ClassN):
        filename = TrainFile.format(label, sample)
        x_train[i, :, :] = io.imread(filename).astype(np.float32)
        t_train[i] = label
        i += 1
i = 0
for label in range(ClassN):
    for sample in range(TestN // ClassN):
        filename = TestFile.format(label, sample)
        x_test[i, :, :] = io.imread(filename).astype(np.float32)
        t_test[i] = label
        i += 1

# 変形
x_train = np.ceil(x_train / 255)
x_test = np.ceil(x_test / 255)
x_train = x_train.reshape((len(x_train), 1, Size, Size))
x_test = x_test.reshape((len(x_test), 1, Size, Size))

# モデル定義
class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            l1=L.Linear(800, 500),
            l2=L.Linear(500, 500),
            l3=L.Linear(500, 500),
            l4=L.Linear(500, 10, initialW=np.zeros(
                (10, 500), dtype=np.float32))
        )

    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = self.l4(h)
        return h

model = CNN()
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習部分
n_epoch = 60
batch_size = 100
for epoch in range(n_epoch):
    sum_loss = 0
    sum_accuracy = 0
    perm = np.random.permutation(TrainN)
    for i in range(0, TrainN, batch_size):
        x = Variable(x_train[perm[i:i+batch_size]])
        t = Variable(t_train[perm[i:i+batch_size]])
        y = model.forward(x)
        model.zerograds()
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data*batch_size
        sum_accuracy += acc.data*batch_size
    print("epoch: {}, mean loss: {}, mean accuracy: {}".format(
        epoch, sum_loss/TrainN, sum_accuracy/TrainN))

# 集計
matrix = np.zeros((ClassN, ClassN))
cnt = 0
for i in range(TestN):
    x = Variable(np.array([x_test[i]], dtype=np.float32))
    t = t_test[i]
    y = model.forward(x)
    y = np.argmax(y.data[0])
    if t == y:
        cnt += 1
    matrix[t, y] += 1
print('confusion matrix')
for t in range(0, ClassN):
    for y in range(0, ClassN):
        print('{:04g}, '.format(matrix[t, y]), end="")
    print()
print('total recognition accuracy: {}'.format(cnt/TestN))
