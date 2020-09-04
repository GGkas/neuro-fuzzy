#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import Preprocessor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import optimizers

class RNN_Model:
	
	batch_size = None
	epochs = None
	embedding_size = None
	network_model = None
	train_input = None
	test_input = None
	preproc = None
	encoder_input_data = None
	decoder_input_data = None
	decoder_target_data = None
	
	def __init__(self, batch_size, epochs, embedding_size, preproc):
		self.batch_size = batch_size
		self,epochs = epochs
		self.embedding_size = embedding_size
		self.train_input = preproc.get_model_train_input()
		self.test_input = preproc.get_model_test_input()
		self.preproc = preproc
	
	def create_encoder_decoder(self, flag):
		k = int(len(self.train_input)/2)
		
		if (flag == 'train'):
			encoder_inp_data = self.train_input[:k, :]
			encoder_inp_data = encoder_inp_data[:, :, np.newaxis]
			decoder_targ_data = self.train_input[k:, :]
			decoder_targ_data = decoder_targ_data[:, :, np.newaxis]
			decoder_inp_data = np.full((k, self.preproc.get_train_dset_len(), 1), -1, dtype='float64')
		
			encoder_inputs = Input(shape=(None, 1))
			encoder_lstm = LSTM(self.embedding_size, return_state=True)
			_, state_h, state_c = encoder_lstm(encoder_inputs)
			encoder_states = [state_h, state_c]
		
			decoder_inputs = Input(shape=(None, 1))
			decoder_lstm = LSTM(self.embedding_size, return_state=True, return_sequences=True)
			decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)
			decoder_dense = Dense(1, activation='linear')
			decoder_outputs = decoder_dense(decoder_outputs)
		
		elif (flag == 'test'):
			encoder_inp_data = self.test_input[:k, :]
			encoder_inp_data = encoder_inp_data[:, :, np.newaxis]
			decoder_targ_data = self.test_input[k:, :]
			decoder_targ_data = decoder_targ_data[:, :, np.newaxis]
			decoder_inp_data = np.full((k, self.preproc.get_test_dset_len(), 1), -1, dtype='float64')
		return encoder_inputs, decoder_inputs, decoder_outputs
	
	def train(self):
		enc_inp, dec_inp, dec_out = create_encoder_decoder(flag='train')
		self.model = Model([enc_inp, dec_inp], dec_out)
		self.model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
		self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data, batch_size=self.batch_size, epochs=self.epochs)
	
	def evaluate(self):
		enc_inp, dec_inp, dec_out = create_encoder_decoder)flag='test')
		self.model.evaluate([enc_inp, dec_inp], dec_out)
	
	def save(self, model_file):
		self.model.save(model_file)
	
