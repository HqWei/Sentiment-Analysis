import numpy as np
import tensorflow as tf
wordVectors = np.load('./training_data/wordVectors.npy')
print ('Loaded the word vectors!')

maxSeqLength = 250
batchSize = 32
numDimensions=50

ids = np.load('./training_data/idsMatrix.npy')
from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append(0)
        else:
            num = randint(13499,24999)
            labels.append(1)
        arr[i] = ids[num-1:num]

    # image = tf.cast(arr, tf.int32)
    # labels = tf.cast(labels, tf.int32)
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append(0)
        else:
            labels.append(1)
        arr[i] = ids[num - 1:num]
    return arr, labels

def get_allTest(step):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = 11499+1+step
        if (num <= 12499):
            labels.append(0)
        else:
            labels.append(1)
        arr[i] = ids[num - 1:num]
    return arr, labels


        # arr, labels=getTrainBatch()
# print(arr.shape)
# print(labels.shape)
#
# input_data = tf.placeholder(tf.int32, [batchSize,maxSeqLength])
# data = tf.Variable(tf.zeros([batchSize,maxSeqLength, numDimensions]),dtype=tf.float32)
# data = tf.nn.embedding_lookup(wordVectors,input_data)
# print(data.shape)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     arr, labels = getTrainBatch();
#     # arr = tf.reshape(arr, [250])
#     da=sess.run(data,{input_data: arr})
#     print(da)
#     print(da.shape)