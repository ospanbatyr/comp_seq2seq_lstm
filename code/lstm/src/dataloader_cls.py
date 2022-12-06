import os
import logging
import pdb
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import unicodedata
from collections import OrderedDict

class TextDataset(Dataset):
	'''
		Expecting csv files with columns ['sent1', 'sent2']

		Args:
						data_path: Root folder Containing all the data
						dataset: Specific Folder==> data_path/dataset/	(Should contain train.csv and dev.csv)
						max_length: Self Explanatory
						is_debug: Load a subset of data for faster testing

	'''

	def __init__(self, data_path='./data', dataset='cogs', datatype='train', max_length=60, is_debug=False, to_sort=False):
		# data_path = '/content/drive/MyDrive/DataLab/turkeyproject/Code/data'
		if datatype=='train':
			file_path = os.path.join(data_path, dataset, 'train.tsv')
		elif datatype=='dev':
			file_path = os.path.join(data_path, dataset, 'dev.tsv')
		elif datatype=='test':
			file_path = os.path.join(data_path, dataset, 'test.tsv')
		else:
			file_path = os.path.join(data_path, dataset, 'gen.tsv')

		self.datatype = datatype

		file_df= pd.read_csv(file_path, sep='\t')

		self.src = file_df['Source'].values
		self.trg = file_df['Input'].values
		self.labels= file_df['Output'].values

		if is_debug:
			self.src = self.src[:5000:500]
			self.trg = self.trg[:5000:500]
			self.labels = self.labels[:5000:500]

		self.max_length = max_length

		all_sents = zip(self.src, self.trg, self.labels)

		if to_sort:
			all_sents = sorted(all_sents, key = lambda x : len(x[0].split()))

		self.src, self.trg, self.labels = zip(*all_sents)

	def __len__(self):
		return len(self.src)

	# TODO, in other datasets, we may need to generalize this function
	def __getitem__(self, idx):
		src = str(self.src[idx])
		trg = str(self.trg[idx])
		labels = int(self.labels[idx])
	
		return {'src': src, 'trg': trg, 'labels': labels, 'idx':idx}

	def curb_to_length(self, string):
		return ' '.join(string.strip().split()[:self.max_length])

	def process_string(self, string):
		#string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " 's", string)
		string = re.sub(r"\'ve", " 've", string)
		string = re.sub(r"n\'t", " n't", string)
		string = re.sub(r"\'re", " 're", string)
		string = re.sub(r"\'d", " 'd", string)
		string = re.sub(r"\'ll", " 'll", string)
		#string = re.sub(r",", " , ", string)
		#string = re.sub(r"!", " ! ", string)
		#string = re.sub(r"\(", " ( ", string)
		#string = re.sub(r"\)", " ) ", string)
		#string = re.sub(r"\?", " ? ", string)
		#string = re.sub(r"\s{2,}", " ", string)
		return string