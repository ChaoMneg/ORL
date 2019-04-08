import numpy as np
import tensorflow as tf
import random,os,cv2,glob
from sklearn.decomposition import PCA
import time
batch_size = 40

def loadImageSet(folder=u'F:\Face_image', sampleCount=5): #加载图像集，随机选择sampleCount张图片用于训练
    trainData = []
    testData = []
    yTrain2=[]
    yTest2=[]
    #print(yTest)
    for k in range(40):
        folder2 = os.path.join(folder, 's%d' % (k+1))  # 拼接路径格式：F:\face_image\s1
        """
        a=glob.glob(os.path.join(folder2, '*.pgm')) # 获取路径下所有的pgm文件
        for d in a:
            #print(d) d=10 每个文件夹有10张图
            img=cv2.imread(d) # imread读取图像返回的是三维数组,返回值是3个数组：I( : , : ,1) I( : , : ,2) I( : , : ,3) 这3个数组中应该存放的是RGB的值
            #print(img) #112*92*3
            cv2.imshow('image', img)
        """
        # data 每次10*112*92
        # cv2.imread()第二个参数为0的时候读入为灰度图，即使原图是彩色图也会转成灰度图
        # glob.glob获取路径下所有的pgm文件，以list的形式返回
        data = [ cv2.imread(d,0) for d in glob.glob(os.path.join(folder2, '*.pgm'))]
        # 在10的范围内随机取sampleCount个数 例：[5, 8, 9, 4, 3]
        sample = random.sample(range(10), sampleCount)
        # ravel将多维数组降为一维数组####原：92*112 现：1*10304
        trainData.extend([data[i].ravel() for i in range(10) if i in sample])
        testData.extend([data[i].ravel() for i in range(10) if i not in sample]) # 40*5列  每列1*10304个元素


        yTrain1 = np.zeros(40) # 生成一个0构成的矩阵--1*40  循环40次
        yTest1 = np.zeros(40)
        yTrain1[k] = 1  # 第K个元素赋值为1
        yTest1[k] = 1
        yTrain = np.tile(yTrain1,5)  # 沿X轴复制yTrain1 5倍,每行200个元素，循环40次
        yTest = np.tile(yTest1,5)
        yTrain2.extend(yTrain) # len(yTrain2) = 40 每列200个元素
        yTest2.extend(yTest)
    """
    trainData : 200列  每列10304个元素 200*10304
    yTrain2 : 40列 每列200个元素  40*200
    testData : 200列  每列10304个元素
    yTest2 : 40列 每列200个元素
    """
    return np.array(trainData),  np.array(yTrain2), np.array(testData), np.array(yTest2)  # 返回数组


"""
添加神经层函数add_layer()
inputs：输入值
in_size：输入有多少个单位
out_size：输出有多少个单位
n_layer：
activation_function：激励函数
"""
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    layer_name="layer%s"%n_layer
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W') # 权重，矩阵in_size*out_size
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b') # 偏差，推荐初始值不为0，所以+0.1
            tf.summary.histogram(layer_name+"/biases",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases) # inputs*Weights+biases ，未被激活的预测值
        if activation_function is None:
            outputs = Wx_plus_b # 如果是线性
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs',outputs)
        return outputs



def main():
    xTrain_, yTrain, xTest_, yTest = loadImageSet()# 对应函数的return值
    yTrain.shape=200,40
    yTest.shape=200,40

    # PCA
    """
    n_components: PCA算法中所要保留的主成分个数n，取大于等于1的整数时，即指定我们希望降维后的维数, 取0-1的浮点数时，即指定降维后的方差和占比，比例越大，保留的信息越多。系统会自行计算保留的维度个数
    copy: 默认为True,表示是否在运行算法时，将原始训练数据复制一份,若为False，则运行PCA算法后，原始训练数据的值会改
    whiten: 白化,默认为False
    
    函数：
    fit_transform(X) 用X来训练PCA模型，同时返回降维后的数据
    inverse_transform() 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)
    transform(X) 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
    """
    pca = PCA(n_components=36) # 200*76
    x_trainPCA = pca.fit_transform(xTrain_)
    x_testPCA = pca.transform(xTest_)

    # print(x_trainPCA.shape)
    # 放置占位符，用于在计算时接收输入值
    with tf.name_scope('inputs'):
        x = tf.placeholder("float", [None, 36],name='x_in')
        y_ = tf.placeholder("float", [None, 40],name='y_in')

    # #定义隐含层，输入神经元个数=特征10304
    # l1 = add_layer(x,10304,1000,n_layer=1 ,activation_function=tf.nn.sigmoid)
    # l2 = add_layer(l1, 1000, 700, n_layer=2, activation_function=tf.nn.sigmoid)
    # l3 = add_layer(l2, 700,200, n_layer=3, activation_function=tf.nn.sigmoid)
    # #定义输出层。此时的输入就是隐藏层的输出——40，输入有110层（隐藏层的输出层），输出有40层
    # y= add_layer(l3,110,40,n_layer=4, activation_function=tf.nn.softmax)

    #定义隐含层，输入神经元个数=特征10304
    l1 = add_layer(x,36,30,n_layer=1 ,activation_function=tf.nn.sigmoid)
    # 定义输出层
    y = add_layer(l1, 30, 40, n_layer=2, activation_function=tf.nn.softmax)


    # 计算交叉墒,计算预测跟实际的差别
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        tf.summary.scalar('loss', cross_entropy )

    # 定义命名空间，使用tensorboard进行可视化
    with tf.name_scope('train'):
        # 使用梯度下降算法以0.01的学习率最小化交叉墒
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 学习效率：小于1的值都可以

    # 开始训练模型，循环1000次，每次都会随机抓取训练数据中的10条数据，然后作为参数替换之前的占位符来运行train_step
    def return_next_batch(batch_size, pos):
        start = pos * batch_size
        end = start + batch_size
        return x_trainPCA[start:end], yTrain[start:end]

    # # 启动初始化，为其分配固定数量的显存GPU
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # 启动初始化
    sess = tf.Session()

    #tf.merge_all_summaries() 方法会对我们所有的 summaries 合并到一起
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', tf.get_default_graph()) # 可视化Tensorboard
    # 初始化之前创建的变量的操作
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        for j in range(5):
            batch_x, batch_y =return_next_batch(batch_size, j)
            # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
            # batch_x/255将图像矩阵值转换为0-1之间
            sess.run(train_step, feed_dict={x: np.matrix( batch_x/255), y_: np.matrix(batch_y)})
            # 输出训练集误差
            print(sess.run(cross_entropy, feed_dict={x: np.matrix((batch_x/255)), y_: np.matrix(batch_y)}))
        if i % 50 == 0:  # 每训练50次，合并一下结果
            result = sess.run(merged, feed_dict={x: np.matrix((batch_x/255)), y_: np.matrix(batch_y)})
            writer.add_summary(result, i)

    # 评估模型，对比预测结果是否跟实际结果一致
    # tf.argmax(y,1)返回y上最大值的索引。因为标签是由0,1组成，返回的索引就是数值为1的位置
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 计算正确预测项的比例，因为tf.equal返回的是布尔值，使用tf.cast可以把布尔值转换成浮点数，tf.reduce_mean是求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 在session中启动accuracy，输入是orl中的测试集
    print(sess.run(accuracy, feed_dict={x:np.matrix(x_testPCA/255), y_: np.matrix(yTest)}))


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
