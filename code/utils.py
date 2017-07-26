
#!/usr/bin/python
# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------------------#
#       author: BinhDT                                                                      #
#       description: preprocess data like exporting aspect, word2vec, load embedding        #
#       prepare data for training                                                           #
#       last update on 14/7/2017                                                            #
#-------------------------------------------------------------------------------------------#

import numpy as np
import os
import re
import csv
from collections import Counter
import codecs
from collections import defaultdict
import xml.etree.ElementTree as ET
import json



def load_embedding(domain, data_dir, flag_addition_corpus, flag_word2vec, flag_use_sentiment_embedding):
    word_dict = dict()
    embedding = list()

    if (flag_addition_corpus):
		# fix it if you want to add more data for word2vec training
		continue
		
    if (flag_word2vec):
        os.system('cd ../fastText && ./fasttext cbow -input ../data/' + domain + '_corpus_for_word2vec.txt -output ../data/' + domain + '_cbow_final -dim 100 -minCount 0 -epoch 2000')
    
    sswe = defaultdict(list)
    if (flag_use_sentiment_embedding):
        f_se = codecs.open('../dictionary/sswe-u.txt', 'r', 'utf-8')
        
        for line in f_se:
            elements = line.split()
            for i in range(1, len(elements)):
                sswe[elements[0].strip()].append(float(elements[i]))
        f_se.close()

    f_vec = codecs.open('../data/' + domain + '_cbow_final_2014.vec', 'r', 'utf-8')
    # f_vec = codecs.open('../data/glove.twitter.27B.100d.txt', 'r', 'utf-8')
    # f_vec = codecs.open('../data/skipgram.wiki.simple.300d.vec', 'r', 'utf-8')
    
    idx = 0
    for line in f_vec:
        if len(line) < 50:
            continue
        else:
            component = line.strip().split(' ')
            word_dict[component[0].lower()] = idx
            if (flag_use_sentiment_embedding and component[0].lower() in sswe.keys()):
                embedding.append(sswe[component[0].lower()])
            else:
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


def load_stop_words():
    stop_words = list()
    fsw = codecs.open('../dictionary/stop_words.txt', 'r', 'utf-8')
    for line in fsw:
        stop_words.append(line.strip())
    fsw.close()
    return stop_words


def load_sentiment_dictionary():
    pos_list = list()
    neg_list = list()
    rev_list = list()
    inc_list = list()
    dec_list = list()
    sent_words_dict = dict()

    fneg = open('../dictionary/negative_words.txt', 'r')
    fpos = open('../dictionary/positive_words.txt', 'r')
    frev = open('../dictionary/reverse_words.txt', 'r')
    fdec = open('../dictionary/decremental_words.txt', 'r')
    finc = open('../dictionary/incremental_words.txt', 'r')

    for line in fpos:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 0
            pos_list.append(line.strip())

    for line in fneg:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 1
            neg_list.append(line.strip())

    for line in frev:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 2
            rev_list.append(line.strip())

    for line in finc:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 3
            inc_list.append(line.strip())

    for line in fdec:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 4
            dec_list.append(line.strip())
            
    fneg.close()
    fpos.close()
    frev.close()
    fdec.close()
    finc.close()

    return pos_list, neg_list, rev_list, inc_list, dec_list, sent_words_dict


def export_aspect(domain, data_dir):
    aspect_list = list()
    
    fa = codecs.open('../dictionary/' + domain + '_aspect.txt', 'w', 'utf-8')
    for file in os.listdir(data_dir):
        if not (file.endswith('.txt') and domain in file):
            continue
            
        f = codecs.open(data_dir + file, 'r', 'utf-8')
        for line in f:
            for word in line.split(' '):
                if '{as' in word:
                    aspect_list.append(word.split('{')[0].strip())
        f.close()
            
    for w in sorted(set(aspect_list)):
        fa.write(w + '\n')
    
    fa.close()
    
    return set(aspect_list)


def sortchildrenby(parent, attr):
    parent[:] = sorted(parent, key=lambda child: int(child.get(attr)))


def change_xml_to_txt_v1(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            sentence = sentences[i].find('text').text
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (end != 0):
                    new_sentence = new_sentence.replace(sentence[start:end],
                                                        sentence[start:end] + '{as' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' unknowntoken{as' + polarity + '}'
                    
            train_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

        except AttributeError:
            continue

    reviews = ET.parse(test_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            sentence = sentences[i].find('text').text
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                if (end != 0):
                    new_sentence = new_sentence.replace(sentence[start:end],
                                                        sentence[start:end] + '{as' + polarity + '}')
                else:
                    new_sentence = new_sentence + ' unknowntoken{as' + polarity + '}'
                    
            test_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

        except AttributeError:
            continue


def change_xml_to_txt_v2(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            sortchildrenby(opinions, 'from')
            bias = 0
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category')
                target = opinion.get('target').lower()
                if (end != 0):
                    new_sentence = new_sentence[:bias+end] + ' ' + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                    bias = bias + len(category + '{as' + polarity + '}') + 1
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'
            train_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

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
            sortchildrenby(opinions, 'from')
            bias = 0
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category')
                if (end != 0):
                    new_sentence = new_sentence[:bias+end] + ' ' + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                    bias = bias + len(category + '{as' + polarity + '}') + 1
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'
                    
            test_text.write(' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n')

        except AttributeError:
            continue


def change_xml_to_txt_v3(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category')
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            train_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'service#general", "service#general").replace("}'s", "}").replace("}'", "}"))

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
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category')
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            test_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'service#general", "service#general").replace("}'s", "}").replace("}'", "}"))

        except AttributeError:
            continue

def change_xml_to_txt_v4(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category').split('#')[0]
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            train_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'service", "service").replace("}'s", "}").replace("}'", "}"))

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
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category').split('#')[0]
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            test_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'service", "service").replace("}'s", "}").replace("}'", "}"))

        except AttributeError:
            continue

def change_xml_to_txt_v5(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    reviews = ET.parse(train_filename).getroot().findall('Review')
    sentences = []
    for r in reviews:
        sentences += r.find('sentences').getchildren()

    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            opinions = sentences[i].find('Opinions').findall('Opinion')
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category').replace('STYLE_OPTIONS', 'style').split('#')[1]
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            train_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'general", "general").replace("}'s", "}").replace("}'", "}"))

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
            sortchildrenby(opinions, 'from')
            bias = 0
            last_start = -1
            last_end = -1
            for opinion in opinions:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                category = opinion.get('category').replace('STYLE_OPTIONS', 'style').split('#')[1]
                target = opinion.get('target').lower()
                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + category + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(category + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + category + '{as' + polarity + '}'

                last_start = start
                last_end = end

            test_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'general", "general").replace("}'s", "}").replace("}'", "}"))

        except AttributeError:
            continue

def change_xml_to_txt_2014(domain, data_dir):
    train_filename = data_dir + domain + '_Train_Final.xml'
    test_filename = data_dir + domain + '_Test.xml'

    train_text = codecs.open(data_dir + domain + '_Train_Final.txt', 'w', 'utf-8')
    test_text = codecs.open(data_dir + domain + '_Test.txt', 'w', 'utf-8')

    sentences = ET.parse(train_filename).getroot()
    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            aspectTerms = sentences[i].find('aspectTerms').findall('aspectTerm')
            sortchildrenby(aspectTerms, 'from')
            bias = 0
            last_start = -1
            last_end = -1

            for opinion in aspectTerms:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                term = re.sub(r'[.,:;/?!\"\n()\\]',' ', opinion.get('term').lower()).strip() 
                if polarity == 'conflict':
                    continue

                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + term + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(term + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + term + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(term + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + term + '{as' + polarity + '}'

                last_start = start
                last_end = end

            train_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace(
                "'bar", "bar").replace(
                "}'s", "}").replace(
                "}'", "}").replace(
                "'pub", "pub").replace(
                "'perks", "perks").replace(
                "'kamasutra", "kamasutra"))

        except AttributeError:
            continue

    sentences = ET.parse(test_filename).getroot()
    for i in range(len(sentences)):
        try:
            new_sentence = sentences[i].find('text').text
            aspectTerms = sentences[i].find('aspectTerms').findall('aspectTerm')
            sortchildrenby(aspectTerms, 'from')
            bias = 0
            last_start = -1
            last_end = -1

            for opinion in aspectTerms:
                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                polarity = opinion.get('polarity')
                term = re.sub(r'[.,:;/?!\"\n()\\]',' ', opinion.get('term').lower()).strip()
                if polarity == 'conflict':
                    continue

                if (end != 0):
                    if (last_start == start and last_end == end):
                        new_sentence = new_sentence[:bias+end+1] + term + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(term + '{as' + polarity + '}') + 1
                    else:
                        new_sentence = new_sentence[:bias+start] + term + '{as' + polarity + '}' + new_sentence[bias+end:]
                        bias = bias + len(term + '{as' + polarity + '}') - (end - start)
                else:
                    new_sentence = new_sentence + ' ' + term + '{as' + polarity + '}'

                last_start = start
                last_end = end
                
            test_text.write((' '.join(re.sub(r'[.,:;/\-?!\"\n()\\]',' ', new_sentence).lower().split()) + '\n').replace("'general", "general").replace("}'s", "}").replace("}'", "}"))

        except AttributeError:
            continue

def load_data(domain, data_dir, flag_word2vec, label_dict, seq_max_len, flag_addition_corpus,
            flag_change_xml_to_txt, negative_weight, positive_weight, neutral_weight, 
            flag_use_sentiment_embedding):
    train_data = list()
    train_mask = list()
    train_binary_mask = list()
    train_label = list()
    train_seq_len = list()
    train_sentiment_for_word = list()
    test_data = list()
    test_mask = list()
    test_binary_mask = list()
    test_label = list()
    test_seq_len = list()
    test_sentiment_for_word = list()
    count_pos = 0
    count_neg = 0
    count_neu = 0

    if (flag_change_xml_to_txt):
        change_xml_to_txt_2014(domain, data_dir)

    stop_words = load_stop_words()
    pos_list, neg_list, rev_list, inc_list, dec_list, sent_words_dict = load_sentiment_dictionary()
    aspect_list = export_aspect(domain, data_dir)
    word_dict, word_dict_rev, embedding = load_embedding(domain, data_dir, flag_addition_corpus, flag_word2vec, flag_use_sentiment_embedding)
    # load data, mask, label
    for file in os.listdir(data_dir):
        if not (file.endswith('.txt') and domain in file):
            continue

        f_processed = codecs.open(data_dir + file, 'r', 'utf-8')
        for line in f_processed:
            data_tmp = list()
            mask_tmp = list()
            binary_mask_tmp = list()
            label_tmp = list()
            sentiment_for_word_tmp = list()
            count_len = 0

            words = line.strip().split(' ')
            for word in words:
                if (word in stop_words):
                    continue
                word_clean = word.replace('{aspositive}', '').replace('{asnegative}', '').replace('{asneutral}', '')

                if (word_clean in word_dict.keys() and count_len < seq_max_len):
                    if (word_clean in pos_list):
                        sentiment_for_word_tmp.append(1)
                    elif (word_clean in neg_list):
                        sentiment_for_word_tmp.append(2)
                    elif (word_clean in rev_list):
                        sentiment_for_word_tmp.append(0)
                    elif (word_clean in inc_list):
                        sentiment_for_word_tmp.append(0)
                    elif (word_clean in dec_list):
                        sentiment_for_word_tmp.append(0)
                    else:
                        sentiment_for_word_tmp.append(0)

                    if ('aspositive' in word):
                        mask_tmp.append(positive_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['aspositive'])
                        count_pos = count_pos + 1
                    elif ('asneutral' in word):
                        mask_tmp.append(neutral_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['asneutral'])
                        count_neu = count_neu + 1
                    elif ('asnegative' in word):
                        mask_tmp.append(negative_weight)
                        binary_mask_tmp.append(1.0)
                        label_tmp.append(label_dict['asnegative'])
                        count_neg = count_neg + 1
                    else:
                        mask_tmp.append(0.)
                        binary_mask_tmp.append(0.)
                        label_tmp.append(0)
                    count_len = count_len + 1

                    data_tmp.append(word_dict[word_clean])
                elif '{as' in word and file != domain + '_Train_Final.txt':
                    print(word)

            if file == domain + '_Train_Final.txt':
                train_seq_len.append(count_len)
            else:
                test_seq_len.append(count_len)

            for _ in range(seq_max_len - count_len):
                data_tmp.append(word_dict['<padding>'])
                mask_tmp.append(0.)
                binary_mask_tmp.append(0.)
                label_tmp.append(0)
                sentiment_for_word_tmp.append(0)

            if file == domain + '_Train_Final.txt':
                train_data.append(data_tmp)
                train_mask.append(mask_tmp)
                train_binary_mask.append(binary_mask_tmp)
                train_label.append(label_tmp)
                train_sentiment_for_word.append(sentiment_for_word_tmp)
            else:
                test_data.append(data_tmp)
                test_mask.append(mask_tmp)
                test_binary_mask.append(binary_mask_tmp)
                test_label.append(label_tmp)
                test_sentiment_for_word.append(sentiment_for_word_tmp)
        f_processed.close()

    print('pos: %d' %count_pos)
    print('neu: %d' %count_neu)
    print('neg: %d' %count_neg)
    print('len of train data is %d' %(len(train_data)))
    print('len of test data is %d' %(len(test_data)))
    data_sample = ''
    for id in train_data[10]:
        data_sample = data_sample + ' ' + word_dict_rev[id]

    print('%s' %data_sample)
    print(train_data[10])
    print(train_mask[10])
    print(train_label[10])
    print(train_sentiment_for_word[10])
    print('len of word dictionary is %d' %(len(word_dict)))
    print('len of embedding is %d' %(len(embedding)))
    print('len of aspect_list is %d' %(len(aspect_list)))
    print('max sequence length is %d' %(np.max(test_seq_len)))

    return train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, \
    test_data, test_mask, test_binary_mask, test_label, test_seq_len, test_sentiment_for_word, \
    word_dict, word_dict_rev, embedding, aspect_list


def main():
    seq_max_len = 60
    negative_weight = 2.5
    positive_weight = 1.0
    neutral_weight = 5.0

    label_dict = {
        'aspositive' : 1,
        'asneutral' : 0,
        'asnegative': 2
    }

    data_dir = '../data/ABSA_SemEval2014/'
    domain = 'Laptops'
    flag_word2vec = False
    flag_addition_corpus = True
    flag_change_xml_to_txt = True
    flag_use_sentiment_embedding = False

    train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, \
    test_data, test_mask, test_binary_mask, test_label, test_seq_len, test_sentiment_for_word, \
    word_dict, word_dict_rev, embedding, aspect_list = load_data(
        domain,
        data_dir,
        flag_word2vec,
        label_dict,
        seq_max_len,
        flag_addition_corpus,
        flag_change_xml_to_txt,
        negative_weight,
        positive_weight,
        neutral_weight,
        flag_use_sentiment_embedding
    )

if __name__ == '__main__':
    main()