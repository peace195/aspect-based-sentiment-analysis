
# coding: utf-8
# tensorflow version 1.2
#---------------------------------------------------------------------------#
#       author: BinhDT                                                      #
#       description: Bi-direction LSTM model for aspect sentiment           # 
#       input: sentences contain aspects                                    #
#       output: sentiment label for aspects                                 #
#       last update on 25/6/2017                                    #
#---------------------------------------------------------------------------#

import json
import codecs
import math
import numpy as np
import utils
import tensorflow as tf
import matplotlib.pyplot as plt

# Read data
batch_size = 128
seq_max_len = 32
nb_sentiment_label = 3
embedding_size = 100
nb_linear_inside = 256
nb_lstm_inside = 256
layers = 1
TRAINING_ITERATIONS = 15000
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0005
negative_weight = 2.0
positive_weight = 1.0
neutral_weight = 3.0

label_dict = {
    'aspositive' : 1,
    'asneutral' : 0,
    'asnegative': 2
}

data_dir = '../data'
flag_word2vec = True
flag_addition_corpus = False
flag_change_file_structure = True
flag_train = True
flag_test = True

train_data, train_mask, train_binary_mask, train_label, \
test_data, test_mask, test_binary_mask, test_label, \
word_dict, word_dict_rev, embedding, aspect_list  = utils.load_data(
    data_dir,
    flag_word2vec,
    label_dict,
    seq_max_len,
    flag_addition_corpus,
    flag_change_file_structure,
    negative_weight,
    positive_weight,
    neutral_weight
)

nb_sample_train = len(train_data)

# Modeling

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    tf_X_train = tf.placeholder(tf.float32, shape=[None, seq_max_len, embedding_size])
    tf_X_train_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
    tf_X_binary_mask = tf.placeholder(tf.float32, shape=[None, seq_max_len])
    tf_y_train = tf.placeholder(tf.int64, shape=[None, seq_max_len])
    keep_prob = tf.placeholder(tf.float32)
    
    ln_w = tf.Variable(tf.truncated_normal([embedding_size, nb_linear_inside], stddev=1.0 / math.sqrt(embedding_size)))
    ln_b = tf.Variable(tf.zeros([nb_linear_inside]))
    
    
    sent_w = tf.Variable(tf.truncated_normal([nb_lstm_inside, nb_sentiment_label],
                                             stddev=1.0 / math.sqrt(2 * nb_lstm_inside)))
    sent_b = tf.Variable(tf.zeros([nb_sentiment_label]))
    

    y_labels = tf.one_hot(tf_y_train, nb_sentiment_label,
                          on_value = 1.0,
                          off_value = 0.0,
                          axis = -1)
     

    X_train = tf.transpose(tf_X_train, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    X_train = tf.reshape(X_train, [-1, embedding_size])
    X_train = tf.add(tf.matmul(X_train, ln_w), ln_b)
    X_train = tf.nn.relu(X_train)
    X_train = tf.split(axis=0, num_or_size_splits=seq_max_len, value=X_train)
    
    # bidirection lstm
    # Creating the forward and backwards cells
    # X_train = tf.stack(X_train)
    # print(X_train.get_shape())
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(nb_lstm_inside, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(nb_lstm_inside, forget_bias=1.0)
    # Pass lstm_fw_cell / lstm_bw_cell directly to tf.nn.bidrectional_rnn
    # if only a single layer is needed
    lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*layers)
    lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*layers)
    # Get lstm cell output
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_multicell,
                                                 lstm_bw_multicell,
                                                 X_train, dtype='float32')
    # output_fw, output_bw = outputs
    # outputs = tf.multiply(output_fw, output_bw)
    # outputs = tf.concat(outputs, 2)
    output_fw, output_bw = tf.split(outputs, [nb_lstm_inside, nb_lstm_inside], 2)
    sentiment = tf.reshape(tf.add(output_fw, output_bw), [-1, nb_lstm_inside]) 
    # sentiment = tf.multiply(sentiment, tf_X_train_mask)
    # sentiment = tf.reduce_mean(sentiment, reduction_indices=1)
    # sentiment = outputs[-1]
    sentiment = tf.nn.dropout(sentiment, keep_prob)
    sentiment = tf.add(tf.matmul(sentiment, sent_w), sent_b)
    sentiment = tf.split(axis=0, num_or_size_splits=seq_max_len, value=sentiment)

    # change back dimension to [batch_size, n_step, n_input]
    sentiment = tf.stack(sentiment)
    sentiment = tf.transpose(sentiment, [1, 0, 2])
    sentiment = tf.multiply(sentiment, tf.expand_dims(tf_X_binary_mask, 2))

    cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=sentiment, labels=y_labels), tf_X_train_mask))
    prediction = tf.argmax(tf.nn.softmax(sentiment), 2)
    correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(prediction, tf_y_train), tf.float32), tf_X_binary_mask))
    # TODO here, fix tf_X_train_mask to 0, 1 vector
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    
    saver = tf.train.Saver()

# Training

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    x_test = list()
    for i in range(len(test_data)):
        sentence = list()
        for word_id in test_data[i]:
            sentence.append(embedding[word_id])
        x_test.append(sentence)
    

    x_train = list()
    for i in range(len(train_data)):
        sentence = list()
        for word_id in train_data[i]:
            sentence.append(embedding[word_id])
        x_train.append(sentence)


    if (flag_train):
        loss_list = list()
        accuracy_list = list()

        for it in range(TRAINING_ITERATIONS):
            #generate batch (x_train, y_train, seq_lengths_train)
            if (it * batch_size % nb_sample_train + batch_size < nb_sample_train):
                index = it * batch_size % nb_sample_train
            else:
                index = nb_sample_train - batch_size

            _, correct_prediction_train, cost_train = sess.run([optimizer, correct_prediction, cross_entropy], 
                                                      feed_dict={tf_X_train: np.asarray(x_train[index : index + batch_size]),
                                                                 tf_X_train_mask: np.asarray(train_mask[index : index + batch_size]),
                                                                 tf_X_binary_mask: np.asarray(train_binary_mask[index : index + batch_size]),
                                                                 tf_y_train: np.asarray(train_label[index : index + batch_size]),
                                                                 keep_prob: 1.0})

            print('training_accuracy => %.3f, cost value => %.5f for step %d, learning_rate => %.5f' % \
                (float(correct_prediction_train)/np.sum(np.asarray(train_binary_mask[index : index + batch_size])), cost_train, it, learning_rate.eval()))
            
            loss_list.append(cost_train)
            accuracy_list.append(float(correct_prediction_train)/np.sum(np.asarray(train_binary_mask[index : index + batch_size])))

            if it % 50 == 0:
                correct_prediction_test, cost_test = sess.run([correct_prediction, cross_entropy], 
                                                  feed_dict={tf_X_train: np.asarray(x_test),
                                                             tf_X_train_mask: np.asarray(test_mask),
                                                             tf_X_binary_mask: np.asarray(test_binary_mask),
                                                             tf_y_train: np.asarray(test_label),
                                                             keep_prob: 1.0})

                print('test accuracy => %.3f , cost value  => %.5f' %(float(correct_prediction_test)/np.sum(test_binary_mask), cost_test))

                plt.plot(accuracy_list)
                axes = plt.gca()
                axes.set_ylim([0, 1.2])
                plt.title('batch train accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('step')
                plt.savefig('accuracy.png')
                plt.close()

                plt.plot(loss_list)
                plt.title('batch train loss')
                plt.ylabel('loss')
                plt.xlabel('step')
                plt.savefig('loss.png')
                plt.close()

        saver.save(sess, '../ckpt/se-apect-v0.ckpt')
    else:
        saver.restore(sess, '../ckpt/se-apect-v0.ckpt')

    correct_prediction_test, prediction_test = sess.run([correct_prediction, prediction], 
                                              feed_dict={tf_X_train: np.asarray(x_test),
                                                         tf_X_train_mask: np.asarray(test_mask),
                                                         tf_X_binary_mask: np.asarray(test_binary_mask),
                                                         tf_y_train: np.asarray(test_label),
                                                         keep_prob: 1.0})


    print('test accuracy => %.3f' %(float(correct_prediction_test)/np.sum(test_binary_mask)))
    f_result = codecs.open('../result/result', 'w', 'utf-8')
    f_result.write('#----------------------------------------------------------------------------------------------#\n')
    f_result.write('#\t author: BinhDT\n')
    f_result.write('#\t test accuracy %.2f\n' %(float(correct_prediction_test)*100/np.sum(np.asarray(test_mask) > 0.)))
    f_result.write(
        '#\t 1:positive, 0:neutral, 2:negative\n')
    f_result.write('#----------------------------------------------------------------------------------------------#\n')

    for i in range(len(test_data)):
        data_sample = ''
        for j in range(len(test_data[i])):
            if word_dict_rev[test_data[i][j]] == '<unk>':
                continue
            elif test_mask[i][j] > 0.:
                data_sample = data_sample + word_dict_rev[test_data[i][j]] + \
                              '(label ' + str(test_label[i][j]) + \
                              '|predict ' + str(prediction_test[i][j]) + ') '
            else:
                data_sample = data_sample + word_dict_rev[test_data[i][j]] + ' '
        f_result.write('%s\n' %data_sample.replace('<padding>', ''))

    f_result.close()