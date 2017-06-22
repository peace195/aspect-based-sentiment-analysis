
#!/usr/bin/python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------------------#
#       author: BinhDT                                                                      #
#       description: preprocess data like exporting aspect, word2vec, load embedding        #
#       prepare data for training                                                           #
#       last update on 11:47PM 23/6/2017                                                    #
#-------------------------------------------------------------------------------------------#

import numpy as np
import os
import csv
from collections import Counter
import codecs
from collections import defaultdict
import xml.etree.ElementTree as ET



def load_embedding(data_dir, flag_addition_corpus, flag_word2vec):
    word_dict = dict()
    embedding = list()
    
    f_corpus = codecs.open('../data/corpus_for_word2vec.txt', 'w', 'utf-8')

    for file in os.listdir(data_dir + '/ABSA_SemEval2015'):
        if file.endswith('.txt'):
            f_processed = codecs.open(data_dir + '/ABSA_SemEval2015' + file, 'r', 'utf-8')
            for line in f_processed:
                corpus = line
                corpus = corpus.replace('{a-positive}', '')
                corpus = corpus.replace('{a-negative}', '')
                corpus = corpus.replace('{a-neutral}', '')
                f_corpus.write(corpus + '\n')

    if (flag_addition_corpus):
        for file in os.listdir(data_dir + '/Addition_Restaurant_Reviews_For_Word2vec'):
            with open(data_dir + '/Addition_Restaurant_Reviews_For_Word2vec/' + file, 'rb') as csvfile:
                if file == '1-restaurant-test.csv':
                    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                    for row in reader:
                        f_corpus.write(row[0].lower().replace('\n', '').replace('"', '') + '\n')
                elif file == '1-restaurant-train.csv':
                    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                    for row in reader:
                        f_corpus.write(row[1].lower().replace('\n', '').replace('"', '') + '\n')
                else:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for row in reader:
                        f_corpus.write(row[9].lower().replace('\n', '').replace('"', '') + '\n')

    f_corpus.close()

    if (flag_word2vec):
        os.system('cd ../fastText && ./fasttext cbow -input ../data/corpus_for_word2vec.txt -output ../data/cbow -dim 256 -minCount 3 -epoch 5')
    
    f_vec = codecs.open('../data/cbow.vec', 'r', 'utf-8')
    idx = 0
    for line in f_vec:
        
        if len(line) < 200:
            continue
        else:
            component = line.strip().split(' ')
            word_dict[component[0].lower()] = idx
            word_vec = list()
            for i in range(1, len(component)):
                word_vec.append(float(component[i]))
            embedding.append(word_vec)
            idx = idx + 1
    f_vec.close()
    word_dict['<padding>'] = idx
    embedding.append([0.] * len(embedding[0]))
    word_dict_rev = {v: k for k, v in word_dict.iteritems()}
    return word_dict, word_dict_rev, embedding


def export_aspect(data_dir):
    aspect_list = list()
    
    fa = codecs.open('../dictionary/aspect.txt', 'w', 'utf-8')
    for file in os.listdir(data_dir + '/ABSA_SemEval2015'):
        if not file.endswith('.txt'):
            continue
            
        f = codecs.open(data_dir + '/ABSA_SemEval2015/' + file, 'r', 'utf-8')
        for line in f:
            for word in line.split(' '):
                if '{a-' in word:
                    aspect_list.append(word.split('{')[0].strip())
        f.close()
            
    for w in sorted(set(aspect_list)):
        fa.write(w + '\n')
    
    ft.close()
    fa.close()
    
    return set(aspect_list)


def change_xml_to_txt(data_dir):
    train_filename = data_dir + '/ABSA_SemEval2015/ABSA-15_Restaurants_Train_Final.xml'
    test_filename = data_dir + '/ABSA_SemEval2015/ABSA15_Restaurants_Test.xml'

    train_text = codecs.open(data_dir + '/ABSA_SemEval2015/ABSA-15_Restaurants_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + '/ABSA_SemEval2015/ABSA15_Restaurants_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (start != 0):
                    new_sentence = new_sentence.replace(new_sentence[start:end],
                                                        new_sentence[start:end].replace(' ', '_') + '{a-' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' unknowntoken{a-' + polarity + '}'
                    
                train_text.write(new_sentence.replace('.', '').replace('!', '').lower() + '\n')

        except AttributeError:
            continue

    reviews = ET.parse(test_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (start != 0):
                    new_sentence = new_sentence.replace(new_sentence[start:end],
                                                        new_sentence[start:end].replace(' ', '_') + '{a-' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' unknowntoken{a-' + polarity + '}'
                    
                test_text.write(new_sentence.replace('.', '').replace('!', '').lower() + '\n')

        except AttributeError:
            continue

def load_data(data_dir, flag_word2vec, label_dict, seq_max_len, flag_addition_corpus,
            flag_change_xml_to_txt, negative_weight, positive_weight, neutral_weight):
    data = list()
    mask = list()
    binary_mask = list()
    label = list()

    count_pos = 0
    count_neg = 0
    count_neu = 0

    if (flag_change_xml_to_txt):
        flag_change_xml_to_txt(data_dir)

    aspect_list = export_aspect(data_dir)
    word_dict, word_dict_rev, embedding = load_embedding(data_dir, flag_addition_corpus, flag_word2vec)

    # load data, mask, label


    for file in os.listdir(data_dir + '/ABSA_SemEval2015/'):
        if not file.endswith('.txt'):
            continue

        f_processed = codecs.open(data_dir + '/ABSA_SemEval2015/' + file, 'r', 'utf-8')
        for line in f_processed:
            data_tmp = list()
            mask_tmp = list()
            binary_mask_tmp = list()
            label_tmp = list()
            count_len = 0

            words = line.split(' ')
            for word in words:
                word_clean = word
                word_clean = word_clean.replace('{a-positive}', '')
                word_clean = word_clean.replace('{a-negative}', '')
                word_clean = word_clean.replace('{a-neutral}', '')
                if (word_clean in word_dict.keys() and count_len < seq_max_len):
                    if ('a-positive' in word):
                        mask_tmp.append(positive_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['a-positive'])
                        count_pos = count_pos + 1
                    elif ('a-neutral' in word):
                        mask_tmp.append(neutral_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['a-neutral'])
                        count_neu = count_neu + 1
                    elif ('a-negative' in word):
                        mask_tmp.append(negative_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['a-negative'])
                        count_neg = count_neg + 1
                    else:
                        mask_tmp.append(0.)
                        binary_mask_tmp.append(0.)
                        label_tmp.append(0)
                    count_len = count_len + 1

                    data_tmp.append(word_dict[word_clean])

            for _ in range(seq_max_len - count_len):
                data_tmp.append(word_dict['<padding>'])
                mask_tmp.append(0.)
                binary_mask_tmp.append(0.)
                label_tmp.append(0)

            data.append(data_tmp)
            mask.append(mask_tmp)
            binary_mask.append(binary_mask_tmp)
            label.append(label_tmp)
        f_processed.close()

    #TODO: get sequence length for each sentence
    print('pos: %d' %count_pos)
    print('neu: %d' %count_neu)
    print('neg: %d' %count_neg)
    print('len of data is %d' %(len(data)))
    data_sample = ''
    for id in data[10]:
        data_sample = data_sample + ' ' + word_dict_rev[id]
    print('%s' %data_sample)
    print(data[10])
    print(mask[10])
    print(label[10])
    print('len of word dictionary is %d' %(len(word_dict)))
    print('len of embedding is %d' %(len(embedding)))
    print('len of aspect_list is %d' %(len(aspect_list)))

    return data, mask, binary_mask, label, word_dict, word_dict_rev, embedding, aspect_list


def main():
    change_xml_to_txt('../data')
    export_aspect('../data')

if __name__ == '__main__':
    main()
