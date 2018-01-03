"""这个文件用于练习使用TensorFlow"""

import numpy as np
import tensorflow as tf

#使用TensorFlow线性回归
def test1():
    x_data=np.random.rand(100).astype(np.float32)
    y_data=x_data*0.1+0.3

    weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
    biases=tf.Variable(tf.zeros([1]))
    y=weight*x_data+biases

    loss=tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train=optimizer.minimize(loss)

    init=tf.global_variables_initializer()

    sess=tf.Session()
    sess.run(init)

    for step in range(200):
        sess.run(train)
        if step%20==0:
            print(step,sess.run(weight),sess.run(biases))

#Session会话控制
def test2():
    matrix1=tf.constant([[3,3]])
    matrix2=tf.constant([[2],[2]])
    product=tf.matmul(matrix1,matrix2)

    #method1
    sess=tf.Session()
    result=sess.run(product)
    print(result)
    sess.close()

    #method 2
    # with tf.Session() as sess:
    #     result=sess.run(product)
    #     print(result)

#Variable 变量
def test3():
    state = tf.Variable(1,name='counter')
    print(state.name)

    one=tf.constant(1)
    new_value=tf.add(state,one)
    update = tf.assign(state,new_value)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(3):
            sess.run(update)
            print(sess.run(state))

#获取多个tensor
def test4():
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2,input3)
    mul = tf.multiply(input1,intermed)

    with tf.Session() as sess:
        result = sess.run([mul,intermed])
        print(result)

#placeholder使用
def test5():
    input1 = tf.placeholder("float",[None,2])
    input2 = tf.placeholder("float",[None,2])
    output = tf.multiply(input1,input2)
    a1=np.zeros([2,2])+3
    a2=np.zeros([2, 2])+2
    with tf.Session() as sess:
        print(sess.run([output],feed_dict={input1:a1,input2:a2}))


if __name__=="__main__":
    test5()
