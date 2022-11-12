from ast import operator
from base64 import decode
import os
from pickle import TRUE
from queue import PriorityQueue
from re import I
import sys
import math
import logging
import pdb
import random
from time import time
import numpy as np
import pandas as pd
from .components.beam_search_node import BeamSearchNode
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from gensim import models
from src.components.encoder import Encoder
from src.components.decoder import DecoderRNN
from src.components.attention import LuongAttnDecoderRNN
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint, bleu_scorer
from src.utils.evaluate import cal_score, stack_to_string
from src.confidence_estimation import *
from collections import OrderedDict
from pprint import pprint

import wandb

class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, device, logger, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.logger = logger

		if self.config.embedding == 'bert':
			self.embedding1 = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding1 = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding1 = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding1  = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding1.weight, -1 * self.config.init_range, self.config.init_range)

		self.logger.debug('Building Encoders...')
		self.encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)

		self.logger.debug('Encoders Built...')
		
		self.fc1 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
		self.out = nn.Linear(self.config.hidden_size // 2, 2)

		self.logger.debug('Fully connected layer initialized ...')

		if self.config.freeze_emb:
			for par in self.embedding1.parameters():
				par.requires_grad = False
		

		# TODO - Check if any of these are set to True in the running script
		"""if self.config.freeze_emb2:
			for par in self.decoder.embedding.parameters():
				par.requires_grad = False
			for par in self.decoder.embedding_dropout.parameters():
				par.requires_grad = False"""
		
		if self.config.freeze_lstm_encoder:
			for par in self.encoder.parameters():
				par.requires_grad = False

		"""if self.config.freeze_lstm_decoder:
			for par in self.decoder.parameters():
				if par not in self.decoder.embedding.parameters() and par not in self.decoder.embedding_dropout.parameters() and par not in self.decoder.out.parameters():
					par.requires_grad = False"""

		"""if self.config.freeze_fc:
			for par in self.decoder.out.parameters():
				par.requires_grad = False"""

		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# As we don't have a decoder that returns log_softmax, we have to use CrossEntropyLoss
		self.criterion = nn.CrossEntropyLoss() 

		self.logger.info('All Model Components Initialized...')

	def _form_embeddings(self, file_path):
		weights_all = models.KeyedVectors.load_word2vec_format(file_path, limit=200000, binary=True)
		weight_req  = torch.randn(self.voc1.nwords, self.config.emb1_size)
		for key, value in self.voc1.id2w.items():
			if value in weights_all:
				weight_req[key] = torch.FloatTensor(weights_all[value])

		return weight_req	

	def _initialize_optimizer(self):
		self.params =   list()
		self.non_emb_params = list()

		if not self.config.freeze_emb:
			self.params = self.params + list(self.embedding1.parameters())

		"""if not self.config.freeze_emb2:
			self.params = self.params + list(self.decoder.embedding.parameters()) + list(self.decoder.embedding_dropout.parameters())
			self.non_emb_params = self.non_emb_params + list(self.decoder.embedding.parameters())"""
		
		if not self.config.freeze_lstm_encoder:
			self.params = self.params + list(self.encoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.encoder.parameters())
		
		"""if not self.config.freeze_lstm_decoder:
			if self.config.use_attn:
				decoder_only_params = list(self.decoder.rnn.parameters()) + list(self.decoder.concat.parameters()) + list(self.decoder.attn.parameters())
			else:
				decoder_only_params = list(self.decoder.rnn.parameters())
			self.params = self.params + decoder_only_params
			self.non_emb_params = self.non_emb_params + decoder_only_params
		"""
		"""if not self.config.freeze_fc:
			self.params = self.params + list(self.decoder.out.parameters())
			self.non_emb_params = self.non_emb_params + list(self.decoder.out.parameters())"""

		if not self.config.freeze_emb:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adamw':
				self.optimizer = optim.AdamW(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.embedding1.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
		else:
			if self.config.opt == 'adam':
				self.optimizer = optim.Adam(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adamw':
				self.optimizer = optim.AdamW(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.non_emb_params, "lr": self.config.lr}]
				)

	def forward(self, input_seq, input_labels, input_len1):
		'''
			Args:
				input_seq1 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len1 (tensor): Length of each sequence in input_len1 | size : [batch_size]
				input_seq2 (tensor): values are word indexes | size : [max_len x batch_size]
				input_len2 (tensor): Length of each sequence in input_len2 | size : [batch_size]
			Returns:
				out (tensor) : Probabilities of each output label for each point | size : [batch_size x num_labels]
		'''

	def trainer(self, src, input_seq, input_labels, input_len1, config, device=None, logger=None):
		'''
			Args:
				src (list): input examples as is (i.e. not indexed) | size : [batch_size]
			Returns:
				
		'''
		self.optimizer.zero_grad()

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq, input_len1 = self.embedding1(src)
			input_seq = input_seq.transpose(0,1)
			# input_seq1: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq, input_len1, self.device)
			# sorted_seqs: Tensor [max_len x BS x emb1_size]
			# input_len1: List [BS]
			# orig_idx: Tensor [BS]
		else:
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq, input_len1, self.device)
			sorted_seqs = self.embedding1(sorted_seqs)


		# print(f"Input seq shape: {input_seq.shape}")
		# print(f"sorted_seqs shape: {sorted_seqs.shape}")

		encoder_outputs, encoder_hidden = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)
		
		# print(f'encoder_outputs.shape: {encoder_outputs.shape}')

		# encoder_outputs = torch.mean(encoder_outputs, 0)

		# print(f'encoder_outputs new shape: {encoder_outputs.shape}')

		h = self.fc1(encoder_outputs) # TODO check dimensions
		# print(f'h.shape: {h.shape}')

		out = self.out(h)

		# print(f'Out.shape: {out.shape}')
		# print(f'Labels.shape: {len(input_labels)}')

		self.loss = self.criterion(out, input_labels)

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)
		self.optimizer.step()

		return out, self.loss.item()


	def evaluate(self, data, input_seq, input_len1):
		src = data['src']
		labels = data['trg']

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq, input_len1 = self.embedding1(src)
			input_seq = input_seq.transpose(0,1)
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq, input_len1, self.device)
		else:
			sorted_seqs, sorted_len, orig_idx = sort_by_len(input_seq, input_len1, self.device)
			sorted_seqs = self.embedding1(sorted_seqs)

		with torch.no_grad():
			encoder_outputs, _ = self.encoder(sorted_seqs, sorted_len, orig_idx, self.device)
			#encoder_outputs = torch.mean(encoder_outputs, 0)
			h = self.fc1(encoder_outputs) # TODO check dimensions
			out = self.out(h)

			loss = self.criterion(out, labels) 

		
		return out, loss


def build_model(config, voc1, device, logger):
	'''
		Add Docstring
	'''
	model = Seq2SeqModel(config, voc1, device, logger)
	model = model.to(device)

	return model



def train_model(model, train_dataloader, val_dataloader, test_dataloader, gen_dataloader, voc1, device, config, logger, min_train_loss=float('inf'), min_val_loss=float('inf'), min_test_loss=float('inf'), 
				min_gen_loss=float('inf'), max_train_acc = 0.0, max_val_acc = 0.0, max_test_acc = 0.0, max_gen_acc = 0.0, best_epoch = 0):
	'''
		Add Docstring
	'''
	
	estop_count=0
	
	for epoch in range(1, config.epochs + 1):
		od = OrderedDict()
		od['Epoch'] = epoch
		print_log(logger, od)

		batch_num = 1
		train_loss_epoch = 0.0
		train_acc_epoch = 0.0
		train_acc_epoch_cnt = 0.0
		train_acc_epoch_tot = 0.0

		start_time= time()
		total_batches = len(train_dataloader)

		for data in train_dataloader:
			src = data['src']

			sent1s = sents_to_idx(voc1, data['src'], config.max_length)
			labels = torch.LongTensor(data['trg'])
			sent1_var, input_len1 = process_batch_cls(sent1s, voc1, device)

			model.train()

			outs, loss = model.trainer(src, sent1_var, labels, input_len1, config, device, logger)
			train_loss_epoch += loss

			wandb.log({"train loss per step": loss})

			if config.show_train_acc:
				model.eval()

				_, preds = torch.max(outs.data, 1)
				
				train_acc_epoch_cnt += (preds == labels).sum().item()
				train_acc_epoch_tot += labels.size(0)

			batch_num += 1
			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

		train_loss_epoch = train_loss_epoch/len(train_dataloader)

		wandb.log({"train loss per epoch": train_loss_epoch})

		if config.show_train_acc:
			train_acc_epoch = train_acc_epoch_cnt/train_acc_epoch_tot
			wandb.log({"train accuracy": train_acc_epoch})
		else:
			train_acc_epoch = 0.0

		time_taken = (time() - start_time)/60.0

		logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))

		if config.dev_set:
			logger.debug('Evaluating on Validation Set:')

		if config.dev_set and (config.dev_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			val_loss_epoch, val_acc_epoch = run_validation(config=config, model=model, dataloader=val_dataloader, disp_tok='DEV', voc1=voc1, device=device, logger=logger, epoch_num = epoch, validation = True)
			wandb.log({"validation loss per epoch": val_loss_epoch})
			wandb.log({"validation accuracy": val_acc_epoch})
		else:
			val_loss_epoch = float('inf')
			val_acc_epoch = 0.0

		if config.test_set:
			logger.debug('Evaluating on Test Set:')

		if config.test_set and (config.test_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			test_loss_epoch, test_acc_epoch = run_validation(config=config, model=model, val_dataloader=test_dataloader, disp_tok='TEST', voc1=voc1, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"test loss per epoch": test_loss_epoch})
			wandb.log({"test accuracy": test_acc_epoch})
		else:
			test_loss_epoch = float('inf')
			test_acc_epoch = 0.0

		if config.gen_set:
			logger.debug('Evaluating on Generalization Set:')

		if config.gen_set and (config.gen_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			gen_loss_epoch, gen_acc_epoch = run_validation(config=config, model=model, val_dataloader=gen_dataloader, disp_tok='GEN', voc1=voc1, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"generalization loss per epoch": gen_loss_epoch})
			wandb.log({"generalization accuracy": gen_acc_epoch})
		else:
			gen_loss_epoch = float('inf')
			gen_acc_epoch = 0.0

		selector_flag = 0

		if train_loss_epoch < min_train_loss:
			min_train_loss = train_loss_epoch

		if train_acc_epoch > max_train_acc:
			max_train_acc = train_acc_epoch
			if config.model_selector_set == 'train':
				selector_flag = 1

		if val_loss_epoch < min_val_loss:
			min_val_loss = val_loss_epoch

		if val_acc_epoch > max_val_acc:
			max_val_acc = val_acc_epoch
			if config.model_selector_set == 'val':
				selector_flag = 1

		if test_loss_epoch < min_test_loss:
			min_test_loss = test_loss_epoch

		if test_acc_epoch > max_test_acc:
			max_test_acc = test_acc_epoch
			if config.model_selector_set == 'test':
				selector_flag = 1

		if gen_loss_epoch < min_gen_loss:
			min_gen_loss = gen_loss_epoch

		if gen_acc_epoch > max_gen_acc:
			max_gen_acc = gen_acc_epoch
			if config.model_selector_set == 'gen':
				selector_flag = 1

		if epoch == 1 or selector_flag == 1:
			best_epoch = epoch

			state = {
				'epoch' : epoch,
				'best_epoch': best_epoch,
				'model_state_dict': model.state_dict(),
				'voc1': model.voc1,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss_epoch' : train_loss_epoch,
				'min_train_loss' : min_train_loss,
				'train_acc_epoch' : train_acc_epoch,
				'max_train_acc' : max_train_acc,
				'val_loss_epoch' : val_loss_epoch,
				'min_val_loss' : min_val_loss,
				'val_acc_epoch' : val_acc_epoch,
				'max_val_acc' : max_val_acc,
				'test_loss_epoch' : test_loss_epoch,
				'min_test_loss' : min_test_loss,
				'test_acc_epoch' : test_acc_epoch,
				'max_test_acc' : max_test_acc,
				'gen_loss_epoch' : gen_loss_epoch,
				'min_gen_loss' : min_gen_loss,
				'gen_acc_epoch' : gen_acc_epoch,
				'max_gen_acc' : max_gen_acc,
			}

			if config.save_model:
				save_checkpoint(state, epoch, logger, config.model_path, config.ckpt)
			estop_count = 0
		else:
			estop_count+=1

		od = OrderedDict()
		od['Epoch'] = epoch
		od['best_epoch'] = best_epoch
		od['train_loss_epoch'] = train_loss_epoch
		od['min_train_loss'] = min_train_loss
		od['val_loss_epoch']= val_loss_epoch
		od['min_val_loss']= min_val_loss
		od['test_loss_epoch']= test_loss_epoch
		od['min_test_loss']= min_test_loss
		od['gen_loss_epoch']= gen_loss_epoch
		od['min_gen_loss']= min_gen_loss
		od['train_acc_epoch'] = train_acc_epoch
		od['max_train_acc'] = max_train_acc
		od['val_acc_epoch'] = val_acc_epoch
		od['max_val_acc'] = max_val_acc
		od['test_acc_epoch'] = test_acc_epoch
		od['max_test_acc'] = max_test_acc
		od['gen_acc_epoch'] = gen_acc_epoch
		od['max_gen_acc'] = max_gen_acc
		print_log(logger, od)

		if estop_count > config.early_stopping:
			logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		store_results(config, max_train_acc, max_val_acc, max_test_acc, max_gen_acc, min_train_loss, min_val_loss, min_test_loss, min_gen_loss, best_epoch)
		logger.info('Scores saved at {}'.format(config.result_path))

	return max_val_acc


def run_validation(config, model, dataloader, disp_tok, voc1, device, logger, epoch_num, validation = False):
	batch_num = 1
	val_loss_epoch = 0.0
	val_acc_epoch = 0.0
	val_acc_epoch_cnt = 0.0
	val_acc_epoch_tot = 0.0

	model.eval()

	refs= []
	hyps= []

	if config.mode == 'test':
		sources, gen_trgs, act_trgs, scores = [], [], [], []

	display_n = config.batch_size

	with open(config.outputs_path + '/outputs.txt', 'w') as f_out:
		pass

	total_batches = len(dataloader)
	for data in dataloader:
		sent1s = sents_to_idx(voc1, data['src'], config.max_length)

		src = data['src']
		labels = data['trg']

		sent1_var, input_len1 = process_batch_cls(sent1s, voc1, device)

		outs, loss = model.evaluate(data, sent1_var, input_len1)			

		_, preds = torch.max(outs.data, 1)
		val_acc_epoch_cnt += (preds == labels).sum().item()
		val_acc_epoch_tot += labels.size(0)

		if config.mode == 'test':
			sources+= data['src']
			act_trgs += labels
			# print(decoder_output)


		if batch_num % config.display_freq == 0:
			for i in range(len(sent1s[:display_n])):
				try:
					od = OrderedDict()
					logger.info('-------------------------------------')
					od['Source'] = ' '.join(sent1s[i])

					od['Target'] = labels[i]

					od['Prediction'] = preds[i]
					print_log(logger, od)
					logger.info('-------------------------------------')
				except:
					logger.warning('Exception: Failed to generate')
					pdb.set_trace()
					break

		val_loss_epoch += loss
		batch_num += 1
		print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)

	if config.mode == 'test':
		results_df = pd.DataFrame([sources, act_trgs, gen_trgs, scores]).transpose()
		results_df.columns = ['Source', 'Actual Target', 'Generated Target', 'Score']
		csv_file_path = os.path.join(config.outputs_path, config.dataset+'.csv')
		results_df.to_csv(csv_file_path, index = False)
		return sum(scores)/len(scores)

	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot

	return val_loss_epoch/(len(dataloader) * config.batch_size), val_acc_epoch


