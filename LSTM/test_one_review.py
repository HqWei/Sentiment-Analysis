#https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
import numpy as np
wordsList = np.load('./training_data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('./training_data/wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)


maxSeqLength = 250

ids = np.load('./training_data/idsMatrix.npy')
from random import randint

def getTestBatch(i1):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        # num = randint(11499,13499)
        num=11499+i+batchSize*i1
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

# def getTestBatch():
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = 11699
#         if (num <= 12499):
#             labels.append([1,0])
#         else:
#             labels.append([0,1])
#         arr[i] = ids[num-1:num]
#     return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
numDimensions = 50 #Dimensions for each word vector
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
#
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)

data = tf.nn.embedding_lookup(wordVectors,input_data)
print("%"*80)
print(data.shape)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

# cell_list = []
# for _ in range(2):
#     lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, state_is_tuple=True)
#     cell_list.append(lstmCell)
#
# lstmCell=tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)

value, h = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])

print("*"*80)
print(int(value.get_shape()[0]) - 1)

last = tf.gather(value, int(value.get_shape()[0]) - 1) #The value of this parameter is 249
prediction = (tf.matmul(last, weight) + bias)
pred=tf.argmax(prediction, 1)
lab= tf.argmax(labels,1)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, './models3/pretrained_lstm.ckpt-80000')
# saver.restore(sess, './training_process_model2/pretrained_lstm.ckpt-379000')
iterations = 83
sum=0
for i in range(iterations):
        nextBatch, nextBatchLabels = getTestBatch(i)
        acc=sess.run([accuracy], {input_data: nextBatch, labels: nextBatchLabels})
        # print(pred)
        # print(lab)
        print(acc)
        sum=sum+acc[0]
        # print(data[1,:,:].shape)
print('Final:%f'%(sum/83))
# '''
#
# '''

        # value, h,data, pred, lab, acc = sess.run([value, h,data, pred, lab, accuracy], {input_data: nextBatch, labels: nextBatchLabels})
        # print(value.shape)
        # print(value[1,1,:])
        # np.savetxt('download_data/sim1.txt', value[1,1,:], fmt='%s', newline='\n')
        # # print(h.shape)
        # print('**'*40)
        # print(h[0])
        # np.savetxt('download_data/review11699.txt', data[1,:,:], fmt='%s', newline='\n')
#########################################################################################
# '''
# The weight and biases
# '''
# weight = sess.run(weight)
# print(weight)
# print(weight.shape)
# np.savetxt('download_data/kernel.txt',weight, fmt='%s', newline='\n')
# bias = sess.run(bias)
# print(bias)
# print(bias.shape)
# np.savetxt('download_data/bias.txt',bias, fmt='%s', newline='\n')
#########################################################################################
