#util_merge_pickles

import pickle
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

def merge_af(mypath):
	pickles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	pickles.sort()

	data_0 = []
	data_1 = []
	for pickle_file in pickles:
		with open(mypath+pickle_file, 'rb') as f:
		    data = pickle.load(f)
		data_0.append(data[0])
		data_1.append(data[1])
		# print(np.asarray(data).shape,pickle_file)

	merged_pickle_data = (np.hstack(data_0),np.hstack(data_1))
	print(np.asarray(merged_pickle_data).shape)

	with open('final_train_rewards.pkl', 'wb') as f_main:
		pickle.dump(merged_pickle_data,f_main)
	

if __name__ == "__merge_af__":
	merge_af()

