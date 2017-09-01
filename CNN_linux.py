# -*- coding: utf-8 -*-
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

# path是大文件夹，里面包含的子文件夹，每一个代表一类别
path = '/home/wangbc1/OCR/Image'
w = 28
h = 28
c = 3

# 读入图像，会将样本和标签打乱顺序返回
def read_img(path):
    cate = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print(folder)
        if folder == '/home/wangbc1/OCR/Image/imageno':
            print('read no')
            a = glob.glob(folder + '/*')
            a = np.array(a)
            num = a.shape[0]
            sequence_oder = np.arange(num)
            np.random.shuffle(sequence_oder)
            a = a[sequence_oder]
            for im in a[0:80000]:
                #print('reading the images:%s' % (im))
                img = cv2.imread(im)
                #img = transform.resize(img, (w, h))
                img = cv2.resize(img,(28,28))
                img = img /255.0
                imgs.append(img)
                labels.append(0)
        else:
            print('read yes')
            for im in glob.glob(folder + '/*'):
                #print('reading the images:%s' % (im))
                img = cv2.imread(im)
                #img = transform.resize(img, (w, h))
                img = cv2.resize(img,(28,28))
                img = img /255.0
                imgs.append(img)
                labels.append(1)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

## 用于把标签转成one_hot标签
# def dense_to_one_hot(labels_dense, num_classes):
#       """Convert class labels from scalars to one-hot vectors."""
#       num_labels = labels_dense.shape[0]
#       index_offset = np.arange(num_labels) * num_classes
#       labels_one_hot = np.zeros((num_labels, num_classes))
#       labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#       return labels_one_hot

if __name__ == '__main__':
    # [data, label] = read_img(path)
    # np.save('/home/wangbc1/OCR/data/data.npy',data)
    # np.save('/home/wangbc1/OCR/data/label.npy', label)
    print ('Loading Image ......')
    data = np.load('/home/wangbc1/OCR/data/data.npy')
    label = np.load('/home/wangbc1/OCR/data/label.npy')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    # -----------------构建网络----------------------
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # 第一个卷积层（28——>14)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(14->7)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(7->3)
    # conv3 = tf.layers.conv2d(
    #     inputs=pool2,
    #     filters=80,
    #     kernel_size=[3, 3],
    #     padding="same",
    #     activation=tf.nn.relu,
    #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    #
    # # 第四个卷积层(12->6)
    # conv4 = tf.layers.conv2d(
    #     inputs=pool3,
    #     filters=128,
    #     kernel_size=[3, 3],
    #     padding="same",
    #     activation=tf.nn.relu,
    #     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                             bias_regularizer = tf.contrib.layers.l2_regularizer(0.003))
    # dense2 = tf.layers.dense(inputs=dense1,
    #                          units=256,
    #                          activation=tf.nn.relu,
    #                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
    #                          bias_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense1,
                             units=2,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # ---------------------------网络结束---------------------------

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_result = tf.cast(tf.argmax(logits, 1), tf.int32)

    # 训练和测试数据，可将n_epoch设置更大一些

    n_epoch = 20
    batch_size = 128
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=30)
    tf.add_to_collection('pred_network', logits)
    max_acc = 0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    noise_factor_list = []
    f = open('/home/wangbc1/OCR/model/22_3/record.txt','w')

    for epoch in range(n_epoch):
        start_time = time.time()

        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print ("   train loss %d: %f" % (epoch+1,(train_loss / n_batch)))
        print ("   train acc %d: %f" % (epoch+1,(train_acc / n_batch)))
        train_loss_list.append(train_loss / n_batch)
        train_acc_list.append(train_acc / n_batch)

        # validation
        val_loss, val_acc, n_batch, nf = 0, 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac, result = sess.run([loss, acc,train_result], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err;
            val_acc += ac;
            n_batch += 1
            noise_factor = [1 for i in range(len(result)) if y_val_a[i] == 0 and result[i] == 1]
            nf += sum(noise_factor)
        print ("   validation loss %d: %f" % (epoch+1,(val_loss / n_batch)))
        print ("   validation acc %d: %f" % (epoch+1,(val_acc / n_batch)))
        print ("   noise factor %d: %f" % (epoch + 1, (nf / (128.0 * n_batch))))
        print ('\n')
        f.write("   train loss %d: %f\n" % (epoch+1,(train_loss / n_batch)))
        f.write("   train acc %d: %f\n" % (epoch+1,(train_acc / n_batch)))
        f.write("   validation loss %d: %f\n" % (epoch+1,(val_loss / n_batch)))
        f.write("   validation acc %d: %f\n" % (epoch+1,(val_acc / n_batch)))
        f.write("   noise factor %d: %f\n\n" % (epoch + 1, (nf / (128.0 * n_batch))))
        val_loss_list.append(val_loss / n_batch)
        val_acc_list.append(val_acc / n_batch)
        noise_factor_list.append(nf / (128 * n_batch))

        if val_acc / n_batch > max_acc:
            max_acc = val_acc / n_batch
            saver.save(sess, '/home/wangbc1/OCR/model/22_3/CNN-Model', global_step=epoch + 1)

    # plt.plot(train_loss_list, 'r', label='train_loss')
    # plt.plot(train_acc_list, 'b', label='train_acc')
    # plt.plot(val_loss_list, 'c', label='val_loss')
    # plt.plot(val_acc_list, 'g', label='val_acc')
    # plt.plot(noise_factor_list, 'y', label='noise_factor')
    # plt.xlabel('epoches')
    # plt.ylabel('loss/acc')
    # plt.ylim(0, 1)
    # plt.xlim(0, n_epoch)
    # plt.title('training progress')
    # plt.legend()
    # plt.show()

    sess.close()
    f.close()