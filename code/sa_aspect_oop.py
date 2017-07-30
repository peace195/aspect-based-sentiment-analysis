
# coding: utf-8
#---------------------------------------------------------------------------#
#       author: BinhDT                                                      #
#       description: Bi-direction LSTM model for aspect sentiment           # 
#       input: sentences contain aspects                                    #
#       output: sentiment label for aspects                                 #
#       last update on 27/7/2017                                    #
#---------------------------------------------------------------------------#

import json
import codecs
import math
import numpy as np
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pylab import *

class Data:
    def __init__(self, domain, data_dir, flag_word2vec, label_dict, seq_max_len, flag_addition_corpus,
     flag_change_file_structure, negative_weight, positive_weight, neutral_weight, flag_use_sentiment_embedding):
        self.seq_max_len = seq_max_len
        self.label_dict = label_dict
        self.domain = domain
        self.data_dir = data_dir
        self.flag_word2vec = flag_word2vec
        self.flag_addition_corpus = flag_addition_corpus
        self.flag_change_file_structure = flag_change_file_structure
        self.negative_weight = negative_weight
        self.positive_weight = positive_weight
        self.neutral_weight = neutral_weight
        self.flag_use_sentiment_embedding = flag_use_sentiment_embedding

        self.train_data, self.train_mask, self.train_binary_mask, self.train_label, self.train_seq_len, self.train_sentiment_for_word, \
        self.test_data, self.test_mask, self.test_binary_mask, self.test_label, self.test_seq_len, self.test_sentiment_for_word, \
        self.word_dict, self.word_dict_rev, self.embedding, aspect_list = utils.load_data(
            self.domain,
            self.data_dir,
            self.flag_word2vec,
            self.label_dict,
            self.seq_max_len,
            self.flag_addition_corpus,
            self.flag_change_file_structure,
            self.negative_weight,
            self.positive_weight,
            self.neutral_weight,
            self.flag_use_sentiment_embedding
        )

        self.nb_sample_train = len(self.train_data)
        
        self.x_test = list()
        for i in range(len(self.test_data)):
            sentence = list()
            for word_id in self.test_data[i]:
                sentence.append(self.embedding[word_id])
            self.x_test.append(sentence)
            
        self.x_train = list()
        for i in range(len(self.train_data)):
            sentence = list()
            for word_id in self.train_data[i]:
                sentence.append(self.embedding[word_id])
            self.x_train.append(sentence)

    def parse_data(self, sentences):
        self.test_data = list()
        self.test_binary_mask = list()

        for sentence in sentences:
            words = sentence.split(' ')
            data_tmp = list()
            binary_mask_tmp = list()
            count_len = 0
            for word in words:
                if (word in word_dict.keys() and count_len < seq_max_len):
                    data_tmp.append(self.word_dict[word])
                    binary_mask_tmp.append(1.0)
                    count_len = count_len + 1
    
            for _ in range(self.seq_max_len - count_len):
                data_tmp.append(word_dict['<padding>'])
                binary_mask_tmp.append(0.)

            self.test_data.append(data_tmp)
            self.test_binary_mask.append(binary_mask_tmp)

        self.x_test = list()
        for i in range(len(self.test_data)):
            sentence = list()
            for word_id in self.test_data[i]:
                sentence.append(self.embedding[word_id])
            self.x_test.append(sentence)

class Model:
    def __init__(self, batch_size, seq_max_len, nb_sentiment_label, nb_sentiment_for_word, embedding_size, nb_linear_inside,
        nb_lstm_inside, layers, TRAINING_ITERATIONS, LEARNING_RATE, WEIGHT_DECAY, flag_train, flag_use_sentiment_for_word, sess):
        self.batch_size = batch_size
        self.seq_max_len = seq_max_len
        self.nb_sentiment_label = nb_sentiment_label
        self.nb_sentiment_for_word = nb_sentiment_for_word
        self.embedding_size = embedding_size
        self.nb_linear_inside = nb_linear_inside
        self.nb_lstm_inside = nb_lstm_inside
        self.layers = layers
        self.TRAINING_ITERATIONS = TRAINING_ITERATIONS
        self.LEARNING_RATE = LEARNING_RATE
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.flag_train = flag_train
        self.flag_use_sentiment_for_word = flag_use_sentiment_for_word
        self.sess = sess

    def modeling(self):
        self.tf_X_train = tf.placeholder(tf.float32, \
            shape = [None, self.seq_max_len, self.embedding_size - int(self.flag_use_sentiment_for_word) * self.nb_sentiment_for_word])
        self.tf_X_sent_for_word = tf.placeholder(tf.int64, shape = [None, self.seq_max_len])
        self.tf_X_train_mask = tf.placeholder(tf.float32, shape = [None, self.seq_max_len])
        self.tf_X_binary_mask = tf.placeholder(tf.float32, shape = [None, self.seq_max_len])
        self.tf_y_train = tf.placeholder(tf.int64, shape = [None, self.seq_max_len])
        self.tf_X_seq_len = tf.placeholder(tf.int64, shape = [None])
        self.keep_prob = tf.placeholder(tf.float32)
        

        self.ln_w = tf.Variable(tf.truncated_normal([self.embedding_size, self.nb_linear_inside], stddev = math.sqrt(3.0 / (self.embedding_size + self.nb_linear_inside))))
        self.ln_b = tf.Variable(tf.zeros([self.nb_linear_inside]))
         
        self.sent_w = tf.Variable(tf.truncated_normal([self.nb_lstm_inside, self.nb_sentiment_label],
                                                 stddev = math.sqrt(3.0 / self.nb_lstm_inside + self.nb_sentiment_label)))
        self.sent_b = tf.Variable(tf.zeros([self.nb_sentiment_label]))

        y_labels = tf.one_hot(self.tf_y_train,
                              self.nb_sentiment_label,
                              on_value = 1.0,
                              off_value = 0.0,
                              axis = -1)
         

        if (self.flag_use_sentiment_for_word):
            X_sent_for_word = tf.one_hot(self.tf_X_sent_for_word, self.nb_sentiment_for_word,
                                 on_value = 20.0,
                                 off_value = 10.0,
                                 axis = -1)

            X_train = tf.concat([self.tf_X_train, X_sent_for_word], 2)
            X_train = tf.transpose(X_train, [1, 0, 2])
        else:
            X_train = tf.transpose(self.tf_X_train, [1, 0, 2])

        # Reshaping to (n_steps * batch_size, n_input)
        X_train = tf.reshape(X_train, [-1, self.embedding_size])
        X_train = tf.nn.relu(tf.add(tf.matmul(X_train, self.ln_w), self.ln_b))
        X_train = tf.nn.dropout(X_train, self.keep_prob)
        X_train = tf.split(axis = 0, num_or_size_splits = self.seq_max_len, value = X_train)
        
        # bidirection lstm
        # Creating the forward and backwards cells
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nb_lstm_inside, forget_bias = 0.8)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nb_lstm_inside, forget_bias = 0.8)
        # Get lstm cell output
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                     lstm_bw_cell,
                                                     X_train,
                                                     dtype='float32')

        output_fw, output_bw = tf.split(outputs, [self.nb_lstm_inside, self.nb_lstm_inside], 2)
        sentiment = tf.reshape(tf.add(output_fw, output_bw), [-1, self.nb_lstm_inside])
        
        # sentiment = tf.reshape(outputs, [-1, 2 * self.nb_lstm_inside]) 
        sentiment = tf.nn.dropout(sentiment, self.keep_prob)
        sentiment = tf.add(tf.matmul(sentiment, self.sent_w), self.sent_b)
        sentiment = tf.split(axis = 0, num_or_size_splits = self.seq_max_len, value = sentiment)

        # Change back dimension to [batch_size, n_step, n_input]
        sentiment = tf.stack(sentiment)
        sentiment = tf.transpose(sentiment, [1, 0, 2])
        sentiment = tf.multiply(sentiment, tf.expand_dims(self.tf_X_binary_mask, 2))

        self.cross_entropy = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits = sentiment, labels = y_labels), self.tf_X_train_mask))
        
        regularization = self.WEIGHT_DECAY * sum(
            tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.cross_entropy = self.cross_entropy + regularization
        
        self.prediction = tf.argmax(tf.nn.softmax(sentiment), 2)
        self.correct_prediction = tf.reduce_sum(tf.multiply(tf.cast(tf.equal(self.prediction, self.tf_y_train), tf.float32), self.tf_X_binary_mask))
        self.global_step = tf.Variable(0, trainable = True)
        self.learning_rate = tf.train.exponential_decay(self.LEARNING_RATE, self.global_step, 1000, 0.65, staircase = True)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy, global_step = self.global_step)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy)

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()


    def load_model(self):
        self.saver.restore(self.sess, '../ckpt/se-apect-v0.ckpt')

    def save_model(self):
        self.saver.save(self.sess, '../ckpt/se-apect-v0.ckpt')

    def predict(self, data):
        prediction_test = self.sess.run(self.prediction, 
                          feed_dict={self.tf_X_train: np.asarray(data.x_test),
                                     self.tf_X_binary_mask: np.asarray(data.test_binary_mask),
                                     self.tf_X_seq_len: np.asarray(data.test_seq_len),
                                     self.tf_X_sent_for_word: np.asarray(data.test_sentiment_for_word),
                                     self.keep_prob: 1.0})


        ret = list()
        for i in range(len(data.test_data)):
                data_sample = ''
                for j in range(len(data.test_data[i])):
                    if data.word_dict_rev[data.test_data[i][j]] == '<unk>':
                        continue
                    elif data.test_binary_mask[i][j] > 0.:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + \
                         '(predict ' + str(prediction_test[i][j]) + ') '
                    else:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + ' '
                ret.append(data_sample.replace('<padding>', '').strip())
        return ret


    def evaluate(self, data, flag_write_to_file, flag_train):
        if (not flag_train):
            self.load_model()

        correct_prediction_test, prediction_test = self.sess.run([self.correct_prediction, self.prediction], 
                                              feed_dict={self.tf_X_train: np.asarray(data.x_test),
                                                         self.tf_X_binary_mask: np.asarray(data.test_binary_mask),
                                                         self.tf_X_seq_len: np.asarray(data.test_seq_len),
                                                         self.tf_X_sent_for_word: np.asarray(data.test_sentiment_for_word),
                                                         self.tf_y_train: np.asarray(data.test_label),
                                                         self.keep_prob: 1.0})

        print('test accuracy => %.3f' %(float(correct_prediction_test)/np.sum(data.test_binary_mask)))

        if float(correct_prediction_test)/np.sum(data.test_binary_mask) > 0.809:
            self.save_model()

        if (flag_write_to_file):
            f_result = codecs.open('../result/result.txt', 'w', 'utf-8')
            f_result.write('#---------------------------------------------------------------------------------------------------------#\n')
            f_result.write('#\t author: BinhDT\n')
            f_result.write('#\t test accuracy %.2f\n' %(float(correct_prediction_test)*100/np.sum(np.asarray(data.test_binary_mask) > 0.)))
            f_result.write('#\t 1:positive, 0:neutral, 2:negative\n')
            f_result.write('#---------------------------------------------------------------------------------------------------------#\n')

            for i in range(len(data.test_data)):
                data_sample = ''
                for j in range(len(data.test_data[i])):
                    if data.word_dict_rev[data.test_data[i][j]] == '<unk>':
                        continue
                    elif data.test_binary_mask[i][j] > 0.:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + '(label ' + str(data.test_label[i][j]) + \
                         '|predict ' + str(prediction_test[i][j]) + ') '
                    else:
                        data_sample = data_sample + data.word_dict_rev[data.test_data[i][j]] + ' '
                f_result.write('%s\n' %data_sample.replace('<padding>', '').strip())

            f_result.close()

    def train(self, data):
        self.sess.run(self.init)
        # self.load_model()
        loss_list = list()
        accuracy_list = list()

        for it in range(self.TRAINING_ITERATIONS):
            # generate batch (x_train, y_train, seq_lengths_train)
            if (it * self.batch_size % data.nb_sample_train + self.batch_size < data.nb_sample_train):
                index = it * self.batch_size % data.nb_sample_train
            else:
                index = data.nb_sample_train - self.batch_size
            

            self.sess.run(self.optimizer, 
                          feed_dict={self.tf_X_train: np.asarray(data.x_train[index : index + self.batch_size]),
                                     self.tf_X_train_mask: np.asarray(data.train_mask[index : index + self.batch_size]),
                                     self.tf_X_binary_mask: np.asarray(data.train_binary_mask[index : index + self.batch_size]),
                                     self.tf_X_seq_len: np.asarray(data.train_seq_len[index : index + self.batch_size]),
                                     self.tf_X_sent_for_word: np.asarray(data.train_sentiment_for_word[index : index + self.batch_size]),
                                     self.tf_y_train: np.asarray(data.train_label[index : index + self.batch_size]),
                                     self.keep_prob: 0.5})

            if it % (len(data.x_train) // self.batch_size) == 0:
                print(it)
                self.evaluate(data, it + 100 >= self.TRAINING_ITERATIONS, self.flag_train)
                '''
                correct_prediction_train, cost_train = self.sess.run([self.correct_prediction, self.cross_entropy], 
                                                  feed_dict={self.tf_X_train: np.asarray(data.x_train),
                                                             self.tf_X_train_mask: np.asarray(data.train_mask),
                                                             self.tf_X_binary_mask: np.asarray(data.train_binary_mask),
                                                             self.tf_X_seq_len: np.asarray(data.train_seq_len),
                                                             self.tf_X_sent_for_word: np.asarray(data.train_sentiment_for_word),
                                                             self.tf_y_train: np.asarray(data.train_label),
                                                             self.keep_prob: 0.8})
                
                print('training_accuracy => %.3f, cost value => %.5f for step %d' % \
                (float(correct_prediction_train)/np.sum(np.asarray(data.train_binary_mask)), cost_train, it))
                
                loss_list.append(cost_train)
                accuracy_list.append(float(correct_prediction_train)/np.sum(np.asarray(data.train_binary_mask)))
                
                _, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.plot(loss_list)
                ax2.plot(accuracy_list, 'r')
                ax1.set_xlabel('epoch')
                ax1.set_ylabel('train loss')
                ax2.set_ylabel('train accuracy')
                ax1.set_title('train accuracy and loss')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.savefig('accuracy_loss.eps', format='eps', dpi=150)
                plt.close()
                '''


        self.sess.close()


def main():
    batch_size = 128
    nb_sentiment_label = 3
    nb_sentiment_for_word = 3
    embedding_size = 100
    nb_lstm_inside = 256
    layers = 1
    TRAINING_ITERATIONS = 5000
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 0.0001
    label_dict = {
        'aspositive' : 1,
        'asneutral' : 0,
        'asnegative': 2
    }
    data_dir = '../data/ABSA_SemEval2014/'
    domain = 'Restaurants'
    if domain == 'Restaurants':
        if '2016' in data_dir:
            seq_max_len = 42
            nb_linear_inside = 128
        elif '2015' in data_dir:
            seq_max_len = 36
            nb_linear_inside = 128
        else:
            seq_max_len = 35
            nb_linear_inside = 512
    else:
        seq_max_len = 42
        nb_linear_inside = 256

    
    flag_word2vec = False
    flag_addition_corpus = False
    flag_change_file_structure = False
    flag_use_sentiment_embedding = False
    flag_use_sentiment_for_word = True
    flag_train = True

    negative_weight = 1.0
    positive_weight = 1.0
    neutral_weight = 1.0

    sess = tf.Session()
    
    data = Data(domain, data_dir, flag_word2vec, label_dict, seq_max_len, flag_addition_corpus,
        flag_change_file_structure, negative_weight, positive_weight, neutral_weight, flag_use_sentiment_embedding)


    if (flag_use_sentiment_for_word):
        embedding_size = embedding_size + nb_sentiment_for_word

    model = Model(batch_size, seq_max_len, nb_sentiment_label, nb_sentiment_for_word, embedding_size, nb_linear_inside,
     nb_lstm_inside, layers, TRAINING_ITERATIONS, LEARNING_RATE, WEIGHT_DECAY, flag_train, flag_use_sentiment_for_word,sess)

    model.modeling()

    if (model.flag_train):
        model.train(data)
    else:
        model.load_model()
        model.evaluate(data, True, model.flag_train)

if __name__ == "__main__":
    main()
