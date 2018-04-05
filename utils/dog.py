import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from merge_pickle import merge_af

import sys
sys.path.append("../")
from reinforce import plot_af


mypath = 'pickle_files/' 						#default path

if((len(sys.argv)==2)):
	mypath = str(sys.argv[1])
merge_af(mypath)

with open('final_train_rewards.pkl', 'rb') as f:
    data = pickle.load(f)
plot_af(data,'test.png')       
print("plot saved.")