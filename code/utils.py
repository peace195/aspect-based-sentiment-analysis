
# coding: utf-8
#-------------------------------------------------------------------------------------------#
#		author: BinhDT																		#
# 		description: preprocess data like exporting term & aspect, word2vec, load embedding #
# 		prepare data for training															#
# 		last update on 11:47PM 07/6/2017													#
#-------------------------------------------------------------------------------------------#

import numpy as np
import os
from collections import Counter
import codecs
from collections import defaultdict

def export_term_aspect(data_dir):
	term_list = list()
	aspect_list = list()
	
	ft = codecs.open('../dictionary/term.txt', 'w', 'utf-8')
	fa = codecs.open('../dictionary/aspect.txt', 'w', 'utf-8')
	for file in os.listdir(data_dir):
		if file.endswith('.txt'):
			continue
		elif file.endswith('.ann'):
			f = codecs.open(data_dir + file, 'r', 'utf-8')
			for line in f:
				if ('term' in line.split('\t')[1] or 't-' in line.split('\t')[1]):
					term_list.append(line.split('\t')[2].strip().lower())
				if ('aspect' in line.split('\t')[1] or 'a-' in line.split('\t')[1]):
					aspect_list.append(line.split('\t')[2].strip().lower())
			f.close()
			
	for w in sorted(set(term_list)):
		ft.write(w + '\n')
	for w in sorted(set(aspect_list)):
		fa.write(w + '\n')
		
	
	ft.close()
	fa.close()
	
	return set(term_list), set(aspect_list)

def load_embedding(data_dir, flag_addition_corpus, flag_word2vec):
	word_dict = dict()
	embedding = list()
	
	f_corpus = codecs.open('../data/corpus_for_word2vec.txt', 'w', 'utf-8')

	for file in os.listdir(data_dir + 'processed/'):
		f_processed = codecs.open(data_dir + 'processed/' + file, 'r', 'utf-8')
		for line in f_processed:
			corpus = line.split('###')[0].strip().lower()
			corpus = corpus.replace('{t-positive}', '')
			corpus = corpus.replace('{a-positive}', '')
			corpus = corpus.replace('{t-negative}', '')
			corpus = corpus.replace('{a-negative}', '')
			corpus = corpus.replace('{t-neutral}', '')
			corpus = corpus.replace('{a-neutral}', '')
			corpus = corpus.replace('{term}', '')
			corpus = corpus.replace('{aspect}', '')
			f_corpus.write(corpus + ' <unk>' + '\n')

	if (flag_addition_corpus):
		# make more corpus for word2vec
		f_add_corpus = codecs.open('../data/addition_corpus/data_segment_final.txt', 'r', 'utf-8')
		# f_add_corpus = codecs.open('../data/addition_corpus/data_segment.text', 'r', 'utf-8')
		for line in f_add_corpus:
			f_corpus.write(line.lower().strip() + ' <unk>' + '\n')
		f_add_corpus.close()
	f_corpus.close()

	if (flag_word2vec):
		os.system('cd ../../fastText && ./fasttext skipgram -input ../Sentiment_Analysis_for_term_aspect/data/corpus_for_word2vec -output ../Sentiment_Analysis_for_term_aspect/data/skipgram -dim 256 -minCount 3 -epoch 5')
	
	f_vec = codecs.open('../data/skipgram.vec', 'r', 'utf-8')
	idx = 0
	for line in f_vec:
		
		if len(line) < 100:
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

# sort index of word ann file incrementaly
def sort_index(data_dir):
	for file in os.listdir(data_dir):
		if file.endswith('.ann'):
			line_list = list()
			f = codecs.open(data_dir + file, 'r', 'utf-8')
			for line in f:
				try:
					tmp = list()
					tmp.append(line.split('\t')[0])
					tmp.append(line.split('\t')[1].split(' ')[0])
					tmp.append(line.split('\t')[1].split(' ')[1])
					tmp.append(line.split('\t')[1].split(' ')[2])
					tmp.append(line.split('\t')[2])
					line_list.append(tmp)
				except IndexError:
					continue
			f.close()

			f = codecs.open(data_dir + file, 'w', 'utf-8')
			line_list.sort(key = lambda x: int(x[2]))
			for i in line_list:
				f.write(i[0] + '\t' + i[1] + ' ' + i[2] + ' ' + i[3] + ' \t' + i[4])

			f.close()

def change_file_structure(data_dir):
	sort_index(data_dir)
	for file in os.listdir(data_dir):
		if file.endswith('.txt'):
			for ann in os.listdir(data_dir):
				if (file.split('.')[0] == ann.split('.')[0] and ann.endswith('.ann')):
					f_txt = codecs.open(data_dir + file, 'r', 'utf-8')
					f_ann = codecs.open(data_dir + ann, 'r', 'utf-8')
					data = f_txt.read()
					inc = 0
					for line in f_ann:
						try:
							a = int(line.split('\t')[1].split(' ')[1]) + inc
							b = int(line.split('\t')[1].split(' ')[2]) + inc
							c = line.split('\t')[1].split(' ')[0]
							inc = inc + len(c) + 2 # note \{ and \} 
							data = data[0:a].lower() + data[a : b].replace(' ', '_').lower() + '{' + c + '}' + data[b : ].lower()
						except ValueError:
							print('fuck ValueError!')
							continue

					f_processed = codecs.open(data_dir + 'processed/processed_' + ann.split('.')[0], 'w', 'utf-8')
					f_processed.write(data)
					f_processed.close()
					f_ann.close()
					f_txt.close()

def load_data(data_dir, flag_word2vec, label_dict, seq_max_len, flag_addition_corpus,
 			flag_change_file_structure, negative_weight, positive_weight, neutral_weight):
	data = list()
	mask = list()
	binary_mask = list()
	label = list()

	count_pos = 0
	count_neg = 0
	count_neu = 0

	term_list, aspect_list = export_term_aspect(data_dir)	
	word_dict, word_dict_rev, embedding = load_embedding(data_dir, flag_addition_corpus, flag_word2vec)

	# load data, mask, label
	if (flag_change_file_structure):
		change_file_structure(data_dir)

	for file in os.listdir(data_dir + 'processed/'):
		f_processed = codecs.open(data_dir + 'processed/' + file, 'r', 'utf-8')
		for line in f_processed:
			data_tmp = list()
			mask_tmp = list()
			binary_mask_tmp = list()
			label_tmp = list()
			count_len = 0
			if not ('negative' in line.split('###')[0] or
				   'positive' in line.split('###')[0] or
				   'neutral' in line.split('###')[0]):
				continue
				
			words = line.split('###')[0].split(' ')
			for word in words:
				word_clean = word
				word_clean = word_clean.replace('{t-positive}', '')
				word_clean = word_clean.replace('{a-positive}', '')
				word_clean = word_clean.replace('{t-negative}', '')
				word_clean = word_clean.replace('{a-negative}', '')
				word_clean = word_clean.replace('{t-neutral}', '')
				word_clean = word_clean.replace('{a-neutral}', '')
				word_clean = word_clean.replace('{term}', '')
				word_clean = word_clean.replace('{aspect}', '')
				if (word_clean in word_dict.keys() and count_len < seq_max_len):
					if ('t-positive' in word):
						mask_tmp.append(positive_weight)
						binary_mask_tmp.append(1.0)
						label_tmp.append(label_dict['t-positive'])
						count_pos = count_pos + 1
					elif ('a-positive' in word):
						mask_tmp.append(positive_weight)
						binary_mask_tmp.append(1.0)
						label_tmp.append(label_dict['a-positive'])
						count_pos = count_pos + 1
					elif ('t-neutral' in word):
						mask_tmp.append(neutral_weight)
						binary_mask_tmp.append(1.0)
						label_tmp.append(label_dict['t-neutral'])
						count_neu = count_neu + 1
					elif ('a-neutral' in word):
						mask_tmp.append(neutral_weight)
						binary_mask_tmp.append(1.0)
						label_tmp.append(label_dict['a-neutral'])
						count_neu = count_neu + 1
					elif ('t-negative' in word):
						mask_tmp.append(negative_weight)
						binary_mask_tmp.append(1.0)
						label_tmp.append(label_dict['t-negative'])
						count_neg = count_neg + 1
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
				#data_tmp.append(word_dict['<unk>'])
				mask_tmp.append(0.)
				binary_mask_tmp.append(0.)
				label_tmp.append(0)
				
			data.append(data_tmp)
			mask.append(mask_tmp)
			binary_mask.append(binary_mask_tmp)
			label.append(label_tmp)
		f_processed.close()


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
	print('len of term_list is %d' %(len(term_list)))
	print('len of aspect_list is %d' %(len(aspect_list)))

	return data, mask, binary_mask, label, word_dict, word_dict_rev, embedding, term_list, aspect_list