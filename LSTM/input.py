import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
wordsList = np.load('./training_data/wordsList.npy')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('./training_data/wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)

p=np.mean(abs(wordVectors))
print(p)
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))

ax0.hist(wordVectors,40,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)

ax0.set_title('pdf')
plt.show()
# _positon = np.argmax(wordVectors)  # get the index of max in the a
# print( _positon)
# m, n = divmod(_positon, 50)
# print ("The raw is ", m)
# print ("The column is ", n)
# print ("The max of the a is ", wordVectors[m, n])

'''
ids = np.load('./training_data/idsMatrix.npy')
print(ids[11699])
print(ids[11699].shape)
# print(ids.shape)
# print(ids[11699].shape)
#
input=ids[11699].reshape([1,250])
print(input.shape)
print(input)
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

maxSeqLength = 250
numDimensions=50
input_data = tf.placeholder(tf.int32, [1,maxSeqLength])
data = tf.Variable(tf.zeros([1,maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
print(data.shape)
print(data)

# sess = tf.InteractiveSession()
with tf.Session() as sess:
    data1=sess.run([data], {input_data: input})
    print('**'*40)
print(data1.shape)
# max=max(data1)
# print(max)
'''