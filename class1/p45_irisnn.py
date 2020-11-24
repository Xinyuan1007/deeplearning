# author: xinyuan time: 23/11/2020.
#利用鸢尾花数据实现前向传播，反向传播，可视化loss曲线

#导入所需模块
from sklearn import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


#导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

#随机打乱数据（因为原始数据有序的，顺序不打乱会影响准确率）
#seed：随机种子，是一个整数，当设置之后，每次生成的随机数都是一样（为方便教学，以保证每个同学的结果一致）
np.random.seed(116)
# 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#将打乱的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不同而报错
x_train= tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签打包，一一对应。（把数据集分批次，每个；批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络参数，4个特征，故输入层为4个输入节点；3个分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结构一样，但实际操作中不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev = 0.1, seed = 1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1 # 学习率为0.1
train_loss_results = [] #将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []#将每轮的acc记录在此列表中，为后续画loss曲线提供数据
epoch = 500 # 循环500次
loss_all = 0 # 每轮分4个step，loss_all 记录四个step生成的四个loss的和

# 训练部分
# 数据集级别的循环，每个epoch循环一次数据集
for epoch in range(epoch):
    #batch级别的循环，每个step循环一个batch
    for step, (x_train, y_train) in enumerate(train_db):
        # with结构记录梯度信息
        with tf.GradientTape() as tape:
            # 神经网络乘加运算
            y = tf.matmul(x_train, w1) + b1
            # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y = tf.nn.softmax(y)
            # 将标签值转换为独热码格式，方便计算loss和accuracy
            y_ = tf.one_hot(y_train, depth=3)
            # 采用均方误差计算损失函数mes = mean（sum（y-out）^2)
            loss = tf.reduce_mean(tf.square(y_ - y))
            # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
            loss_all += loss.numpy()
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss,[w1,b1])
        # 实现梯度更新，w1=w1-lr*w1_grad  b1=b1-lr*b1_grad
        w1.assign_sub(lr*grads[0]) # w1 自更新
        b1.assign_sub(lr*grads[1]) # b1 自更新

    # 每个epoch，打印loss信息
    print("Epoch{}, loss: {}".format(epoch,loss_all/4))
    #将4个step的loss求平均记录在此变量中
    train_loss_results.append(loss_all/4)
    #loss_all 归零，为记录下一个epoch的loss做准备
    loss_all = 0

    #测试部分
    # total_correct为预测对的样本个数，
    # total_number 为测试的总样本个数，将这两个变量初始化为0
    total_correct, total_number = 0,0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        # 返回y中最大值的索引，，即预测的分类
        pred = tf.argmax(y, axis = 1)
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype = y_test.dtype)
        # 若分类正确，则correct = 1，否则为0， 将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype= tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]

    # 总准确率等于total_correct/total_number
    acc = total_correct/total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()



