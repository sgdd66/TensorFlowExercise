"""使用TensorFlow学习MNIST数据集"""


import struct
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.preprocessing as pp

def load_mnist():
    image_train_path="C:\\Users\\Administrator\\Documents\\GitHub\\TensorFlowExercise\\train-images.idx3-ubyte"
    label_train_path="C:\\Users\\Administrator\\Documents\\GitHub\\TensorFlowExercise\\train-labels.idx1-ubyte"
    image_test_path="C:\\Users\\Administrator\\Documents\\GitHub\\TensorFlowExercise\\t10k-images.idx3-ubyte"
    label_test_path="C:\\Users\\Administrator\\Documents\\GitHub\\TensorFlowExercise\\t10k-labels.idx1-ubyte"

    with open(label_train_path,'rb') as file1:
        magic,n = struct.unpack('>II',file1.read(8))
        labels_train = np.fromfile(file1,dtype=np.uint8)

    with open(image_train_path,'rb') as file2:
        magic,num,rows,cols = struct.unpack('>IIII',file2.read(16))
        images_train = np.fromfile(file2,dtype=np.uint8).reshape(len(labels_train),784)

    with open(label_test_path,'rb') as file3:
        magic , n = struct.unpack('>II', file3.read(8))
        labels_test = np.fromfile(file3,dtype=np.uint8)

    with open(image_test_path,'rb') as file4:
        magic,num,rows,cols = struct.unpack('>IIII',file4.read(16))
        images_test = np.fromfile(file4,dtype=np.uint8).reshape(len(labels_test),784)

    return images_test,labels_test,images_train,labels_train

def showData():
    images_test, labels_test, images_train, labels_train=load_mnist()

    print(images_train.shape)
    # enc = pp.OneHotEncoder()
    # enc.fit(labels_test)
    # data = enc.transform(labels_test).toarray()
    data = pp.OneHotEncoder().fit_transform(labels_test.reshape(-1,1))
    print(data.shape)

    fig,ax=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)

    ax=ax.flatten()
    for i in range(25):
        ax[i].imshow(images_train[labels_train==5][i].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()

#使用单层神经网络与softmax激励函数处理MNIST问题
def softmax():
    #prepare data
    images_test, labels_test, images_train, labels_train = load_mnist()
    images_test=images_test/255
    images_train=images_train/255
    labels_test = pp.OneHotEncoder().fit_transform(labels_test.reshape(-1, 1)).toarray()
    labels_train = pp.OneHotEncoder().fit_transform(labels_train.reshape(-1,1)).toarray()

    #compute model
    x=tf.placeholder("float", [None, 784])
    w=tf.Variable(tf.random_uniform([784,10],minval=0.01,maxval=0.02))
    b = tf.Variable(tf.zeros([10]))
    tem=tf.matmul(x,w)+b
    y=tf.nn.softmax(tf.matmul(x,w)+b)
    y_=tf.placeholder("float",[None,10])
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))

    #train
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(y,feed_dict={x:images_test,y_:labels_test}))
        for i in range(1000):
            batch_xs=images_train[(i%600)*100:((i%600)*100+100)]
            batch_ys=labels_train[(i%600)*100:((i%600)*100+100)]
            sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

        #evaluation
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        print(sess.run(accuracy, feed_dict={x: images_test, y_: labels_test}))

#使用CNN处理MNIST问题
class CNNforMNIST(object):
    def __init__(self):
        # data
        images_test, labels_test, images_train, labels_train = load_mnist()
        images_test = images_test / 255
        images_train = images_train / 255
        labels_test = pp.OneHotEncoder().fit_transform(labels_test.reshape(-1, 1)).toarray()
        labels_train = pp.OneHotEncoder().fit_transform(labels_train.reshape(-1, 1)).toarray()

        # compute model

        x = tf.placeholder("float",[None,784])
        y_ = tf.placeholder("float",[None,10])

        #first layer
        w_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_variable([32])

        x_image = tf.reshape(x,[-1,28,28,1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image,w_conv1)+b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        #second layer
        w_conv2 = self.weight_variable([5,5,32,64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,w_conv2)+b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        #fully link
        w_fc1 = self.weight_variable([7*7*64,1024])
        b_fc1 = self.weight_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

        #dropout
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

        #output layer
        w_fc2 = self.weight_variable([1024,10])
        b_fc2 = self.bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

        #aim
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

        #train
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        #evaluate
        correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(1000):
                batch_xs = images_train[(i % 600) * 100:((i % 600) * 100 + 100)]
                batch_ys = labels_train[(i % 600) * 100:((i % 600) * 100 + 100)]
                sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys, keep_prob:0.5})
                if i%100==0:
                    train_accuracy= accuracy.eval(feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1})
                    print("the {0}th train outcome:{1:.2f}".format(i,train_accuracy))
            print("the test outcome:{:.2f}".format(sess.run(accuracy,feed_dict={x:images_test, y_:labels_test, keep_prob:1})))






    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




if __name__=='__main__':
    # showData()
    a=CNNforMNIST()


