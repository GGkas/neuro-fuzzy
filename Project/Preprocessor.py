#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##### START OF PREPROCESSOR MODULE #####

#Importing necassary packages/libraries
import numpy as np
import xlrd
import sys
import sleep
from keras.preprocessing.text import Tokenizer

class Preprocessor:
	
	locality = None
	factor = None
	train_memrefs_arr = None
	test_memrefs_arr = None
	train_probs_arr = None
	test_probs_arr = None
	dataset = None
	train_dset = None
	test_dset = None
	
	
	def __init__(self, locality, t_factor):
		self.locality = locality
		self.factor = t_factor
	
	def get_key(val, my_dict):
		for key, value in my_dict.items():
			if (val == value):
				return key
		
		return -1
	
	def get_val(val, my_dict):
		for key, value in my_dict.items():
			if val == key:
				return value
		
		return -1
	
	def times_exists(val, my_vect):
		count = 0
		for i in range(len(my_vect)):
			if (val == my_vect[i]):
				count += 1
		
		return count
	
	def create_dataset(self, filename):
		self.dataset = []
		try:
			wb = xlrd.open_workbook(filename)
			sheet = wb.sheet_by_index(0)
			sheet.cell_value(0, 0)
			
			for i in range(int(sheet.nrows/10)):
				self.dataset.append(sheet.cell_value(i, 0))
		except Exception as err:
			print("Unexcpected error:", sys.exc_info()[0])
			sys.exit(-1)
		finally:
			dataset = np.array(dataset)
			if(dataset.size == 0):
				print("Cannot work with an empty dataset! Exiting...")
				sleep(0.5)
				sys.exit(-1)
			train_size = int(len(dataset)*self.factor)
			test_size = len(dataset) - train_size
			train = dataset[:train_size]
			test = dataset[train_size:]
			
			tr = Tokenizer()
			tr.fit_on_texts(train)
			self.train_dataset = tr.word_index
			trainset_call = tr.word_counts
			
			t = Tokenizer()
			t.fit_on_texts(test)
			self.test_dataset = t.word_index
			testet_call = t.word_counts
			
			self.train_memrefs_arr = np.zeros((int(train_size/self.locality), self.locality))
			self.test_memrefs_arr = np.zeros((int(test_size/self.locality), self.locality))
			
			k=0
			for i in range(int(train_size/self.locality)):
				for j in range(self.locality):
					if(get_val(train[j+k], self.train_dataset) != -1):
						self.train_memrefs_arr[i][j] = get_val(train[j+k], self.train_dataset)
				k += self.locality
			k=0
			for i in range(int(test_size/self.locality)):
				for j in range(self.locality):
					if(get_val(test[j+k], self.test_dataset) != -1):
						self.test_memrefs_arr[i][j] = get_val(test[j+k], self.test_dataset)
				k += self.locality
			
			self.train_probs_arr = np.zeros((int(train_size/self.locality), len(self.train_dataset)))
			self.test_probs_arr = np.zeros((int(test_size/self.locality), len(self.test_dataset)))
			
			for i in range(int(train_size/self.locality)):
				for j in range(len(self.train_dataset):
					   self.train_probs_arr[i][j] = times_exists((j+1), self.train_memrefs_arr[i])/self.locality
			for i in range(int(test_size/self.locality)):
				for j in range(len(self.test_dataset):
					   self.test_probs_arr[i][j] = times_exists((j+1), self.test_memrefs_arr[i])/self.locality
	
	def get_model_train_input(self):
		return self.train_probs_arr
	def get_model_test_input(self):
		return self.test_probs_arr
	def get_train_dset_len(self):
		return len(self.train_dataset)
	def get_test_dset_len(self):
		return len(self.test_dataset)
