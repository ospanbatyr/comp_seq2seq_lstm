import os
from pickle import TRUE
from re import I
import pdb
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from gensim import models
from src.components.encoder import Encoder
from src.components.contextual_embeddings import BertEncoder, RobertaEncoder
from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint
from src.confidence_estimation import *
from collections import OrderedDict
from pprint import pprint
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

class Seq2SeqModel(nn.Module):
	def __init__(self, config, voc1, voc2, device, logger, EOS_tag='</s>', SOS_tag='<s>'):
		super(Seq2SeqModel, self).__init__()

		self.config = config
		self.device = device
		self.voc1 = voc1
		self.voc2 = voc2
		self.EOS_tag = EOS_tag
		self.SOS_tag = SOS_tag
		self.logger = logger

		self.norm0 = nn.LayerNorm(self.config.emb1_size)

		if self.config.embedding == 'bert':
			self.embedding_src = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding_src = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding_src = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding_src = nn.Embedding(self.voc1.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding_src.weight, -1 * self.config.init_range, self.config.init_range)

		# TODO - In the future, if we want to give different embeddings to different encoders, we need to change the following block
		if self.config.embedding == 'bert':
			self.embedding_trg = BertEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'roberta':
			self.embedding_trg = RobertaEncoder(self.config.emb_name, self.device, self.config.freeze_emb)
		elif self.config.embedding == 'word2vec':
			self.config.emb1_size = 300
			self.embedding_trg = nn.Embedding.from_pretrained(torch.FloatTensor(self._form_embeddings(self.config.word2vec_bin)), freeze = self.config.freeze_emb)
		else:
			self.embedding_trg = nn.Embedding(self.voc2.nwords, self.config.emb1_size)
			nn.init.uniform_(self.embedding_trg.weight, -1 * self.config.init_range, self.config.init_range)

		self.logger.debug('Building Encoders...')
		# TODO - In the future, we may want to change emb1
		self.source_encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)

		self.target_encoder = Encoder(
			self.config.hidden_size,
			self.config.emb1_size,
			self.config.cell_type,
			self.config.depth,
			self.config.dropout,
			self.config.bidirectional
		)

		self.logger.debug('Encoders Built...')
		
		self.norm1 = nn.LayerNorm(self.config.hidden_size * 2)
		self.fc1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
		self.relu = nn.ReLU()
		self.norm2 = nn.LayerNorm(self.config.hidden_size)
		self.fc2 = nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
		self.norm3 = nn.LayerNorm(self.config.hidden_size // 2)
		self.out = nn.Linear(self.config.hidden_size // 2, 2)

		self.logger.debug('Fully connected layers initialized ...')

		if self.config.freeze_emb:
			for par in self.embedding_src.parameters():
				par.requires_grad = False
		
		if self.config.freeze_emb:
			for par in self.embedding_trg.parameters():
				par.requires_grad = False

		# TODO - Check if any of these are set to True in the running script
		"""if self.config.freeze_emb2:
			for par in self.decoder.embedding.parameters():
				par.requires_grad = False
			for par in self.decoder.embedding_dropout.parameters():
				par.requires_grad = False"""
		
		if self.config.freeze_lstm_encoder:
			for par in self.target_encoder.parameters():
				par.requires_grad = False
			for par in self.source_encoder.parameters():
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

	# TODO - Change this as it only forms embeddings for emb1
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
			self.params = self.params + list(self.embedding_src.parameters())
			self.params = self.params + list(self.embedding_trg.parameters())

		"""if not self.config.freeze_emb2:
			self.params = self.params + list(self.decoder.embedding.parameters()) + list(self.decoder.embedding_dropout.parameters())
			self.non_emb_params = self.non_emb_params + list(self.decoder.embedding.parameters())"""
		
		if not self.config.freeze_lstm_encoder:
			self.params = self.params + list(self.target_encoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.target_encoder.parameters())
			self.params = self.params + list(self.source_encoder.parameters())
			self.non_emb_params = self.non_emb_params + list(self.source_encoder.parameters())
		
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
					[{"params": self.embedding_src.parameters(), "lr": self.config.emb_lr},
					{"params": self.embedding_trg.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adamw':
				self.optimizer = optim.AdamW(
					[{"params": self.embedding_src.parameters(), "lr": self.config.emb_lr},
					{"params": self.embedding_trg.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'adadelta':
				self.optimizer = optim.Adadelta(
					[{"params": self.embedding_src.parameters(), "lr": self.config.emb_lr},
					{"params": self.embedding_trg.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			elif self.config.opt == 'asgd':
				self.optimizer = optim.ASGD(
					[{"params": self.embedding_src.parameters(), "lr": self.config.emb_lr},
					{"params": self.embedding_trg.parameters(), "lr": self.config.emb_lr},
					{"params": self.non_emb_params, "lr": self.config.lr}]
				)
			else:
				self.optimizer = optim.SGD(
					[{"params": self.embedding_src.parameters(), "lr": self.config.emb_lr},
					{"params": self.embedding_trg.parameters(), "lr": self.config.emb_lr},
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
		
		self.scheduler = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=15)
	
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

	def trainer(self, source, target, input_seq_src, input_seq_trg, input_labels, input_len_src, input_len_trg, config, device=None, logger=None):
		'''
			Args:
				src (list): input examples as is (i.e. not indexed) | size : [batch_size]
			Returns:
				
		'''
		self.optimizer.zero_grad()

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq_src, input_len_src = self.embedding_src(source)
			input_seq_src = input_seq_src.transpose(0,1)
			sorted_seqs_src, sorted_len_src, orig_idx_src = sort_by_len(input_seq_src, input_len_src, self.device)

			input_seq_trg, input_len_trg = self.embedding_trg(target)
			input_seq_trg = input_seq_trg.transpose(0,1)
			sorted_seqs_trg, sorted_len_trg, orig_idx_trg = sort_by_len(input_seq_trg, input_len_trg, self.device)

		else:
			sorted_seqs_src, sorted_len_src, orig_idx_src = sort_by_len(input_seq_src, input_len_src, self.device)
			sorted_seqs_src = self.embedding_src(sorted_seqs_src)
			sorted_seqs_src = self.norm0(sorted_seqs_src) # embedding norm

			sorted_seqs_trg, sorted_len_trg, orig_idx_trg = sort_by_len(input_seq_trg, input_len_trg, self.device)
			sorted_seqs_trg = self.embedding_trg(sorted_seqs_trg)
			sorted_seqs_src = self.norm0(sorted_seqs_src) # embedding norm


		# print(f"Input seq shape: {input_seq.shape}")
		# print(f"sorted_seqs shape: {sorted_seqs.shape}")

		encoder_src_outputs, encoder_src_hidden = self.source_encoder(sorted_seqs_src, sorted_len_src, orig_idx_src, self.device)
		encoder_trg_outputs, encoder_trg_hidden = self.target_encoder(sorted_seqs_trg, sorted_len_trg, orig_idx_trg, self.device)

		# print(f'encoder_outputs.shape: {encoder_outputs.shape}')

		# print(f'encoder_outputs new shape: {encoder_outputs.shape}')

		encoder_outputs = torch.cat((encoder_src_outputs, encoder_trg_outputs), 1)
		
		encoder_outputs = self.norm1(encoder_outputs)
		h = self.fc1(encoder_outputs) # TODO check dimensions
		# print(f'h.shape: {h.shape}')

		h = self.relu(h)

		h = self.norm2(h)
		h = self.fc2(h)
		h = self.relu(h)

		h = self.norm3(h)
		out = self.out(h)

		if device is not None:
			self.loss = self.criterion(out, input_labels.to(device))
		else:
			self.loss = self.criterion(out, input_labels)

		self.loss.backward()
		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.params, self.config.max_grad_norm)

		self.optimizer.step()

		return out.detach().cpu(), self.loss.item()


	def evaluate(self, data, input_seq_src, input_seq_trg, input_len_src, input_len_trg):
		src = data['src']
		trg = data['trg']
		labels = data['labels']
		idx = data['idx']

		if self.config.embedding == 'bert' or self.config.embedding == 'roberta':
			input_seq_src, input_len_src = self.embedding_src(src)
			input_seq_src = input_seq_src.transpose(0,1)
			sorted_seqs_src, sorted_len_src, orig_idx_src = sort_by_len(input_seq_src, input_len_src, self.device)

			input_seq_trg, input_len_trg = self.embedding_trg(trg)
			input_seq_trg = input_seq_trg.transpose(0,1)
			sorted_seqs_trg, sorted_len_trg, orig_idx_trg = sort_by_len(input_seq_trg, input_len_trg, self.device)
		else:
			sorted_seqs_src, sorted_len_src, orig_idx_src = sort_by_len(input_seq_src, input_len_src, self.device)
			sorted_seqs_src = self.embedding_src(sorted_seqs_src)
			sorted_seqs_src = self.norm0(sorted_seqs_src) # embedding norm

			sorted_seqs_trg, sorted_len_trg, orig_idx_trg = sort_by_len(input_seq_trg, input_len_trg, self.device)
			sorted_seqs_trg = self.embedding_trg(sorted_seqs_trg)
			sorted_seqs_src = self.norm0(sorted_seqs_src) # embedding norm

		with torch.no_grad():
			encoder_outputs_src, _ = self.source_encoder(sorted_seqs_src, sorted_len_src, orig_idx_src, self.device)
			encoder_outputs_trg, _ = self.target_encoder(sorted_seqs_trg, sorted_len_trg, orig_idx_trg, self.device)

			encoder_outputs = torch.cat((encoder_outputs_src, encoder_outputs_trg), 1)

			h = self.fc1(encoder_outputs) # TODO check dimensions
			h = self.relu(h) # TODO check dimensions
			h = self.fc2(h)
			h = self.relu(h)
			out = self.out(h)
			
			if self.device is not None:
				loss = self.criterion(out, labels.to(self.device))
			else:
				loss = self.criterion(out, labels)
		
		return out.detach().cpu(), loss.item()


def build_model(config, voc1, voc2, device, logger):
	'''
		Add Docstring
	'''
	model = Seq2SeqModel(config, voc1, voc2, device, logger)
	model = model.to(device)

	return model

#cartography{
def save_tensor(tensor, directory, file_name, epoch):
	
	path = 'outputs_folder/cartographyOut'
	os.makedirs(path, exist_ok=True)
	file_name = os.path.join(path, f"epoch{epoch}_{file_name}_{list(tensor.shape)}.pt")
	print(f"Saving {tensor.shape} to \"{file_name}\"")
	torch.save(tensor.cpu(), file_name)

def extract_after_epoch_logits():
	after_epoch_train_idxs = []
	after_epoch_train_logits = []
	model.eval()
	for step, inputs in tqdm(enumerate(epoch_iterator), desc=f"after epoch[{epoch}] logits", total=len(epoch_iterator)):
		inputs = self._prepare_inputs(inputs)
		after_epoch_train_idxs.extend(inputs["idx"])
		with torch.no_grad():
			loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
			after_epoch_train_logits.append(outputs.logits.clone().detach().cpu())
	model.train()
	save_tensor(torch.cat(after_epoch_train_logits), "training_dynamics_after_epoch", "after_epoch_train_logits")
	save_tensor(torch.stack(after_epoch_train_idxs), "training_dynamics_after_epoch", "after_epoch_train_idxs")

#cartography}	

def train_model(model, train_dataloader, val_dataloader, test_dataloader, gen_dataloader, voc1, voc2, device, config, logger, min_train_loss=float('inf'), min_val_loss=float('inf'), min_test_loss=float('inf'), 
				min_gen_loss=float('inf'), max_train_acc = 0.0, max_val_acc = 0.0, max_test_acc = 0.0, max_gen_acc = 0.0, max_train_auc = 0.0, max_val_auc = 0.0, max_test_auc = 0.0, max_gen_auc = 0.0, best_epoch = 0):
	'''
		Add Docstring
	'''
	estop_count=0
	
	#cartography{
	columns = ['src', 'target', 'epoch', 'label', 'idx']
	df = pd.DataFrame(columns=columns) 
	#cartography
	
	for epoch in range(1, config.epochs + 1):

        #cartography{
		train_logits = []
		train_idxs = []
		train_labels = []
		train_src = []
		train_target = []
		epoch_index = []
		epoch_label = []
        #cartography}

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

		y_scores = []
		y = []
		for data in train_dataloader:
			src = data['src']
			trg = data['trg']
			idx = data['idx']

			sent_srcs = sents_to_idx(voc1, data['src'], config.max_length)
			sent_trgs = sents_to_idx(voc2, data['trg'], config.max_length)

			labels = torch.LongTensor(data['labels'])
			sent_src_var, sent_trg_var, input_len_src, input_len_trg = process_batch_cls(sent_srcs, sent_trgs, voc1, voc2, device)

			model.train()


			outs, loss = model.trainer(src, trg, sent_src_var, sent_trg_var, labels, input_len_src, input_len_trg, config, device, logger)
			train_loss_epoch += loss

            #cartography{
			train_logits.append(outs.data)
			train_idxs.extend(idx)
			train_labels.extend(labels)
			train_src.append(src)
			train_target.append(trg)
	    #cartography}


			y_scores.append(outs.data[:, 1])
			y.append(labels)

			wandb.log({"train loss per step": loss})

			if config.show_train_acc:
				model.eval()

				_, preds = torch.max(outs.data, 1)
				
				train_acc_epoch_cnt += (preds == labels).sum().item()
				train_acc_epoch_tot += labels.size(0)

			batch_num += 1
			print("Completed {} / {}...".format(batch_num, total_batches), end = '\r', flush = True)



		#cartography

		epoch_lst = [epoch]*len(trg)
		# print(len(src))
		# print(len(trg))
		# print(len(epoch_lst))
		# print(len(data['idx']))
		# print(len(data['labels']))
		# print(len(train_logits))

		new_df = pd.DataFrame(   
			{'src': src,
     		'target': trg,
     		'label': data['labels'],
    		'epoch':epoch_lst,
    		'idx': data['idx'].tolist()
    		}, columns =columns) 	

		df = df.append(new_df, ignore_index=True)
		#cartography
		#cartography
		save_tensor(torch.cat(train_logits), "training_dynamics", "train_logits", epoch)
		save_tensor(torch.stack(train_idxs), "training_dynamics", "train_idxs", epoch)
		save_tensor(torch.stack(train_labels), "training_dynamics", "train_labels", epoch)

		df.to_excel("/content/drive/MyDrive/DataLab/turkeyproject/Code/outputs/output.xlsx") 
		#print("---------------------------------training dynamics saved---------------------------------------")

		print("---------------------------------training dynamics saved---------------------------------------")
		#cartography

		train_loss_epoch = train_loss_epoch/len(train_dataloader)
		y_scores, y = torch.cat(y_scores, dim=0), torch.cat(y, dim=0)
		train_auc_epoch = roc_auc_score(y, y_scores)

		
		model.scheduler.step(train_loss_epoch) # just to overfit

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
			val_loss_epoch, val_acc_epoch, val_auc_epoch = run_validation(config=config, model=model, dataloader=val_dataloader, disp_tok='DEV', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = True)
			
			#model.scheduler.step(val_loss_epoch)
			wandb.log({"validation loss per epoch": val_loss_epoch})
			wandb.log({"validation accuracy": val_acc_epoch})
			wandb.log({"validation roc auc score": val_auc_epoch})
		else:
			val_loss_epoch = float('inf')
			val_acc_epoch = 0.0
			val_auc_epoch = 0.0

		if config.test_set:
			logger.debug('Evaluating on Test Set:')

		if config.test_set and (config.test_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			test_loss_epoch, test_acc_epoch, test_auc_epoch = run_validation(config=config, model=model, val_dataloader=test_dataloader, disp_tok='TEST', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"test loss per epoch": test_loss_epoch})
			wandb.log({"test accuracy": test_acc_epoch})
			wandb.log({"test roc auc score": test_auc_epoch})
		else:
			test_loss_epoch = float('inf')
			test_acc_epoch = 0.0
			test_auc_epoch = 0.0

		if config.gen_set:
			logger.debug('Evaluating on Generalization Set:')

		if config.gen_set and (config.gen_always or epoch >= config.epochs - (config.eval_last_n - 1)):
			gen_loss_epoch, gen_acc_epoch, gen_auc_epoch = run_validation(config=config, model=model, val_dataloader=gen_dataloader, disp_tok='GEN', voc1=voc1, voc2=voc2, device=device, logger=logger, epoch_num = epoch, validation = False)
			wandb.log({"generalization loss per epoch": gen_loss_epoch})
			wandb.log({"generalization accuracy": gen_acc_epoch})
			wandb.log({"gen roc auc score": gen_auc_epoch})
		else:
			gen_loss_epoch = float('inf')
			gen_acc_epoch = 0.0
			gen_auc_epoch = 0.0

		selector_flag = 0

		if train_loss_epoch < min_train_loss:
			min_train_loss = train_loss_epoch

		if train_acc_epoch > max_train_acc:
			max_train_acc = train_acc_epoch
			if config.model_selector_set == 'train':
				selector_flag = 1

		if train_auc_epoch > max_train_auc:
			max_train_auc = train_auc_epoch
		
		if val_auc_epoch > max_val_auc:
			max_val_auc = val_auc_epoch

		if test_auc_epoch > max_test_auc:
			max_test_auc = test_auc_epoch
		
		if gen_auc_epoch > max_gen_auc:
			max_gen_auc = gen_auc_epoch

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
				'voc2': model.voc2,
				'optimizer_state_dict': model.optimizer.state_dict(),
				'train_loss_epoch' : train_loss_epoch,
				'min_train_loss' : min_train_loss,
				'train_acc_epoch' : train_acc_epoch,
				'train_auc_epoch': train_auc_epoch,
				'max_train_acc' : max_train_acc,
				'max_train_auc' : max_train_auc,
				'val_loss_epoch' : val_loss_epoch,
				'min_val_loss' : min_val_loss,
				'val_acc_epoch' : val_acc_epoch,
				'val_auc_epoch': val_auc_epoch,
				'max_val_auc' : max_val_auc,
				'max_val_acc' : max_val_acc,
				'test_loss_epoch' : test_loss_epoch,
				'min_test_loss' : min_test_loss,
				'test_acc_epoch' : test_acc_epoch,
				'test_auc_epoch': test_auc_epoch,
				'max_test_auc' : max_test_auc,
				'max_test_acc' : max_test_acc,
				'gen_loss_epoch' : gen_loss_epoch,
				'min_gen_loss' : min_gen_loss,
				'gen_acc_epoch' : gen_acc_epoch,
				'gen_auc_epoch': gen_auc_epoch,
				'max_gen_acc' : max_gen_acc,
				'max_gen_auc' : max_gen_auc,
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
		od['val_auc_epoch'] = val_auc_epoch
		od['train_auc_epoch'] = train_auc_epoch
		od['test_auc_epoch'] = test_auc_epoch
		od['gen_auc_epoch'] = gen_auc_epoch
		od['max_train_auc'] = max_train_auc
		od['max_val_auc'] = max_val_auc
		od['max_test_auc'] = max_test_auc
		od['max_gen_auc'] = max_gen_auc
		print_log(logger, od)

		if estop_count > config.early_stopping:
			logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			break

	logger.info('Training Completed for {} epochs'.format(config.epochs))

	if config.results:
		pass
		#store_results(config, max_train_acc, max_val_acc, max_test_acc, max_gen_acc, min_train_loss, min_val_loss, min_test_loss, min_gen_loss, best_epoch)
		#logger.info('Scores saved at {}'.format(config.result_path))

	return max_val_acc


def run_validation(config, model, dataloader, disp_tok, voc1, voc2, device, logger, epoch_num, validation = False):
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

	y_scores = []
	y = []

	for data in dataloader:
		sent_srcs = sents_to_idx(voc1, data['src'], config.max_length)
		sent_trgs = sents_to_idx(voc2, data['trg'], config.max_length)

		src = data['src']
		trg = data['trg']
		labels = data['labels']

		sent_src_var, sent_trg_var, input_len_src, input_len_trg = process_batch_cls(sent_srcs, sent_trgs, voc1, voc2, device)

		outs, loss = model.evaluate(data, sent_src_var, sent_trg_var, input_len_src, input_len_trg)			

		_, preds = torch.max(outs.data, 1)
		val_acc_epoch_cnt += (preds == labels).sum().item()
		val_acc_epoch_tot += labels.size(0)
		
		y_scores.append(outs.data[:, 1])
		y.append(labels)

		if config.mode == 'test':
			sources+= data['src']
			act_trgs += data['trg']
			# print(decoder_output)


		if batch_num % config.display_freq == 0:
			for i in range(len(sent_srcs[:display_n])):
				try:
					od = OrderedDict()
					logger.info('-------------------------------------')
					od['Source'] = ' '.join(sent_srcs[i])

					od['Target'] = ' '.join(sent_trgs[i])
					
					od['Label'] = labels[i]
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

	y_scores, y = torch.cat(y_scores, dim=0), torch.cat(y, dim=0)
	val_acc_epoch = val_acc_epoch_cnt/val_acc_epoch_tot
	val_auc_epoch = roc_auc_score(y, y_scores)

	

	return val_loss_epoch/(len(dataloader) * config.batch_size), val_acc_epoch, val_auc_epoch


