#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import input_text_data
import text_model2
from input_text_data import batchSize

N_CLASSES = 2
IMG_H = 250
IMG_W = 50
# BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 100000
learning_rate = 0.0001
maxSeqLength=250
numDimensions=50
wordVectors = np.load('./training_data/wordVectors.npy')

logs_train_dir='./text_log5_2_4'
def run_training():
    input_data = tf.placeholder(tf.int32, [batchSize ,maxSeqLength])
    labels = tf.placeholder(tf.int32, [batchSize])

    # data = tf.Variable(tf.zeros([batchSize ,maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)
    data=tf.reshape(data,[batchSize ,maxSeqLength, numDimensions,1])

    train_logits = text_model2.inference4(data, batchSize, N_CLASSES)
    train_loss = text_model2.losses(train_logits, labels)
    train_op = text_model2.trainning(train_loss, learning_rate)
    train_acc = text_model2.evaluation(train_logits, labels)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()


    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './text_log5_2_4/model.ckpt-16000')

        for i in range(MAX_STEP):
            train_batch, train_label_batch = input_text_data.getTrainBatch()
            _, tra_loss, tra_acc =sess.run([train_op, train_loss, train_acc],{input_data: train_batch,labels:train_label_batch})

            if i % 100 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (i, tra_loss, tra_acc))

                test_batch, test_label_batch = input_text_data.getTestBatch()
                test_loss, test_acc = sess.run([train_loss, train_acc],
                                         {input_data: test_batch, labels: test_label_batch})
                print("*****************, test loss = %.2f, test accuracy = %.2f%%" % (test_loss, test_acc))

            #     summary_str = sess.run(summary_op)
            #     train_writer.add_summary(summary_str, i)
            if i % 2000 == 0 or (i + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, "model1.ckpt")
                saver.save(sess, checkpoint_path, global_step=i)
# def get_one_text():

def evaluate_text():
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    labels = tf.placeholder(tf.int32, [batchSize])

    # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)
    data = tf.reshape(data, [batchSize, maxSeqLength, numDimensions, 1])

    train_logits = text_model2.inference(data, batchSize, N_CLASSES)
    train_loss = text_model2.losses(train_logits, labels)
    # train_op = text_model.trainning(train_loss, learning_rate)
    train_acc = text_model2.evaluation(train_logits, labels)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, './text_log/model.ckpt-29999')

        for i in range(MAX_STEP):
            train_batch, train_label_batch = input_text_data.getTestBatch()
            tra_loss, tra_acc = sess.run([ train_loss, train_acc],
                                            {input_data: train_batch, labels: train_label_batch})

            if i % 100 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (i, tra_loss, tra_acc))

def get_exam_acc():
    #set batSize=1
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    labels = tf.placeholder(tf.int32, [batchSize])

    # data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)
    data = tf.reshape(data, [batchSize, maxSeqLength, numDimensions, 1])

    train_logits = text_model2.inference(data, batchSize, N_CLASSES)
    train_loss = text_model2.losses(train_logits, labels)
    # train_op = text_model.trainning(train_loss, learning_rate)
    train_acc = text_model2.evaluation(train_logits, labels)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    all_accuracy=0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './text_log5_2_2/model.ckpt-20000')

        for i in range(2000):
            train_batch, train_label_batch = input_text_data.get_allTest(i)
            tra_loss, tra_acc = sess.run([train_loss, train_acc],
                                         {input_data: train_batch, labels: train_label_batch})
            all_accuracy=all_accuracy+tra_acc
        print('All_accuracy:')
        print(all_accuracy/2000)
            # if i % 100 == 0:
            #     print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (i, tra_loss, tra_acc))


# run_training()
# evaluate_text()
get_exam_acc()