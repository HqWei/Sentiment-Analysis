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

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels



batchSize = 24
lstmUnits = 64
numClasses = 2
numDimensions = 50 #Dimensions for each word vector
iterations = 380000

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
#############################################################
lstmCell= tf.contrib.rnn.BasicLSTMCell(lstmUnits, state_is_tuple=True)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
##########################################################################
# cell_list = []
# for _ in range(2):
#     lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, state_is_tuple=True)
#     cell_list.append(lstmCell)
# lstmCell=tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True)
###############################################################################


initial_state = lstmCell.zero_state(batchSize, tf.float32)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, initial_state=initial_state,dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
tf.summary.scalar('weights1', weight[1,1])
tf.summary.scalar('weights2', weight[3,1])
tf.summary.histogram("weights",weight)

bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
tf.summary.histogram("biases",bias)
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

import datetime
sess = tf.InteractiveSession()
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
# logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
logdir="./mygraph/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './models3/pretrained_lstm.ckpt-80000')

    for i in range(300000,iterations):
       #Next Batch of reviews
       nextBatch, nextBatchLabels = getTrainBatch();
       sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

       #Write summary to Tensorboard
       if (i % 50 == 0):
           summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
           writer.add_summary(summary, i)
           l,acc= sess.run([loss,accuracy], {input_data: nextBatch, labels: nextBatchLabels})
           print('training! Step%d Training loss: %f  Training acc:%f'%(i,l,acc))

       #Save the network every 10,000 training iterations
       if (i % 10000 == 0 and i != 0):
           save_path = saver.save(sess, "./training_process_model2/pretrained_lstm.ckpt", global_step=i)
           print("saved to %s" % save_path)
    writer.close()



