#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
from gym.wrappers import Monitor
import collections
import time
import math
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env, replay, deep, duel):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.learning_rate = 0.001																							#HYPERPARAMETER1

		if(deep==False and duel==False): 
			print("Setting up linear network....")
			self.model = Sequential()
			# self.model.add(Dense(env.action_space.n, input_dim = env.observation_space.shape[0], activation='linear', kernel_initializer='he_uniform', use_bias = True))
			self.model.add(Dense(24, input_dim = env.observation_space.shape[0], activation='linear', kernel_initializer='he_uniform',use_bias = True))
			self.model.add(Dense(env.action_space.n, input_dim = 24, activation='linear', kernel_initializer='he_uniform',use_bias = True))
			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='graphs/Linear.png', show_shapes = True)
			self.model.summary()
		
		elif(deep==True):	
			print("Setting up DDQN network....")
			self.model = Sequential()
			self.model.add(Dense(24, input_dim = env.observation_space.shape[0], activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			self.model.add(Dense(24, input_dim = 24,activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			# self.model.add(Dropout(0.5))
			self.model.add(Dense(24, input_dim = 24, activation='relu', kernel_initializer='he_uniform',use_bias = True))
			# self.model.add(BatchNormalization())
			self.model.add(Dense(env.action_space.n, input_dim = 24, activation='linear', kernel_initializer='he_uniform',use_bias = True))
			print("Q-Network initialized.... :)\n")

			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='graphs/DDQN.png', show_shapes = True)
			self.model.summary()

		elif(duel==True):			
			print("Setting up Dueling DDQN network....")
			inp = Input(shape=(env.observation_space.shape[0],))
			layer_shared1 = Dense(24,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
			# layer_shared1 = BatchNormalization()(layer_shared1)
			layer_shared2 = Dense(24,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared1)
			# layer_shared2 = BatchNormalization()(layer_shared2)
			print("Shared layers initialized....")

			layer_v1 = Dense(24,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
			# layer_v1 = BatchNormalization()(layer_v1)
			layer_a1 = Dense(24,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_shared2)
			# layer_a1 = BatchNormalization()(layer_a1)
			layer_v2 = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_v1)
			layer_a2 = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_a1)
			print("Value and Advantage Layers initialised....")

			layer_mean = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(layer_a2)
			temp = layer_v2
			temp2 = layer_mean

			for i in range(env.action_space.n-1):
				layer_v2 = keras.layers.concatenate([layer_v2,temp],axis=-1)
				layer_mean = keras.layers.concatenate([layer_mean,temp2],axis=-1)
				
			# layer_q = Lambda(lambda x: K.expand_dims(x[0],axis=-1)  + x[1] - K.mean(x[1],keepdims=True), output_shape=(env.action_space.n,))([layer_v2, layer_a2])
			layer_q = Subtract()([layer_a2,layer_mean])
			layer_q = Add()([layer_q,layer_v2])

			print("Q-function layer initialized.... :)\n")

			self.model = Model(inp, layer_q)
			self.model.summary()
			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='graphs/Duel_DQN.png', show_shapes = True)
		# elif(space==True):			

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights.
		self.model.save_weights(suffix)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		self.model = keras.models.load_model(model_file)

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		# self.model.load_weights(weight_file)
		self.model.set_weights(weight_file)

	def visualise_weights(self):
		print("Current Weights\n")
		for layer in self.model.layers:
			temp = layer.get_weights()
			print(temp)


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions.
		self.burn_in = burn_in
		self.memory_size = memory_size
		self.experience = collections.deque()
		self.batch = []

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
		# You will feed this to your model to train.
		indices = np.random.randint(0,len(self.experience),batch_size)
		self.batch = [self.experience[i] for i in indices]


	def append(self, transition):
		# Appends transition to the memory.
		if(len(self.experience)>self.memory_size):
			pop = self.experience.popleft()
		self.experience.append(transition)


class DQN_Agent():

	# In this class, we will implement functions to do the following.
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy.
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.

	def __init__(self, env, replay, deep, duel, render, env_name):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.net = QNetwork(env,replay,deep,duel)
		self.prediction_net = QNetwork(env,replay,deep,duel)
		self.replay = replay
		self.deep = deep
		self.duel = duel
		self.env = env
		self.env_name = env_name
		self.replay_mem = Replay_Memory()																	   #HYPERPARAMETER2
		self.render = render
		self.feature_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n
		self.discount_factor = 1 	

		if(env_name == "CartPole-v0"):
			self.discount_factor = 0.99
		elif(env_name == "MountainCar-v0"):
			self.discount_factor = 1

		self.train_iters = 100000
		self.epsilon = 0.5 																						#HYPERPARAMETER3
		self.epsilon_min = 0.05																					#HYPERPARAMETER4
		self.num_episodes = 4000
		self.epsilon_decay = float((self.epsilon-self.epsilon_min)/100000)										#HYPERPARAMETER5
		self.update_prediction_net_iters =500 																	#HYPERPARAMETER6
		self.avg_rew_buf_size_epi = 10 
		self.save_weights_iters = 10000
		self.save_model_iters = 10000 													
		self.print_epi = 1 
		self.print_loss_epi = 50 
		self.save_vid = int(self.num_episodes/4)

		self.evaluate = 0.0

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.
		if(np.random.random_sample()<self.epsilon):
			return np.random.randint(self.action_size)
		else:
			return np.argmax(q_values[0])

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time.
		return np.argmax(q_values[0])
		

	def train(self):
		# In this function, we will train our network.
		# If training without experience replay_memory, then you will interact with the environment
		# in this function, while also updating your network parameters.
		folder = 'none'
		test_reward = []
		train_reward = []
		curr_episode = 1
		iters = 1
		max_reward = 0
		reward_buf = collections.deque()
		
		self.burn_in_memory()
		print("burnin done")	

		save_episode_id=np.zeros(1)

		if(self.env_name=="CartPole-v0"):
			save_episode_id=np.around(np.linspace(0,40000,num=41))
		elif(self.env_name=="MountainCar-v0"):
			save_episode_id=np.around(np.linspace(0,self.num_episodes,num=4))

		if(self.duel):
			video_file_path = 'videos/'+str(self.env_name)+'/duel/'
		if(self.deep):
			video_file_path = 'videos/'+str(self.env_name)+'/deep/'
		if(self.replay):
			video_file_path = 'videos/'+str(self.env_name)+'/replay/'
		else:
			video_file_path = 'videos/'+str(self.env_name)+'/linear/'

		self.env = Monitor(self.env,video_file_path,video_callable= lambda episode_id: episode_id in save_episode_id, force=True)

		complete = 0
		# while(iters<self.train_iters): 																			#uncomment for cartpole
		for e in range(self.num_episodes):																	#uncomment for mountaincar
			curr_reward = 0
			curr_iters = 0
			# print("Episode resets")
			curr_state = self.env.reset()
			curr_state = curr_state.reshape([1,self.feature_size])
			curr_action = self.epsilon_greedy_policy(self.net.model.predict(curr_state))
			# print("Inside new episode")
			# while(iters<self.train_iters): 																		#uncomment for cartpole
			while(True): 																						#uncomment for mountaincar
				# self.env.render()
				if(self.replay==False and self.deep==False and self.duel==False):

					nextstate, reward, is_terminal, _ = self.env.step(curr_action)
					curr_reward += pow(self.discount_factor,curr_iters)*reward
					# truth = np.zeros(shape=[1,self.action_size])

					if(is_terminal == True):
						if(self.env_name == "CartPole-v0"):
							complete+=1
							q_target = reward
							truth = self.net.model.predict(curr_state)
							truth[0][curr_action] = q_target
							self.net.model.fit(curr_state,truth,epochs=1,verbose=0)
						elif(self.env_name == "MountainCar-v0" and nextstate[0]>0.5):
							complete+=1
							q_target = reward
							truth = self.net.model.predict(curr_state)
							truth[0][curr_action] = q_target
							self.net.model.fit(curr_state,truth,epochs=1,verbose=0)
						break

					nextstate = nextstate.reshape([1,self.feature_size])
					q_nextstate = self.net.model.predict(nextstate)

					q_target = reward + self.discount_factor*np.amax(q_nextstate)
					truth = self.net.model.predict(curr_state)
					truth[0][curr_action] = q_target

					if(curr_episode%self.print_loss_epi==0):
						self.net.model.fit(curr_state,truth,epochs=1,verbose=1)
					else:
						self.net.model.fit(curr_state,truth,epochs=1,verbose=0)

					nextaction = self.epsilon_greedy_policy(q_nextstate)
					curr_state = nextstate
					curr_action = nextaction

					iters += 1
					curr_iters += 1

					# if(iters%self.save_weights_iters==0):
					# 	self.net.save_model_weights(backup)
					if(iters%self.save_model_iters==0):
						if(self.env_name == "CartPole-v0"):
							self.net.model.save('models/linear/cartpole/'+ str(iters) +'_cp_BN_linear_nrp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
							folder = 'linear/cartpole'
						elif(self.env_name == "MountainCar-v0"):
							self.net.model.save('models/linear/mountaincar/'+ str(iters) +'_mc_BN_linear_nrp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
							folder = 'linear/mountaincar'
				


				else:
					nextstate, reward, is_terminal, _ = self.env.step(curr_action)
					self.replay_mem.append([curr_state,curr_action,reward,nextstate,is_terminal])

					curr_reward += pow(self.discount_factor,curr_iters)*reward
					if(is_terminal):
						complete+=1
						break
					# print(len(self.replay_mem.experience))
					self.replay_mem.sample_batch()
					
					input_state = np.zeros(shape=[len(self.replay_mem.batch),self.feature_size])
					truth = np.zeros(shape=[len(self.replay_mem.batch),self.action_size])
					
					for i in range(len(self.replay_mem.batch)):
						state_t,action_t,reward_t,nextstate_t,_ = self.replay_mem.batch[i]

						nextstate_t = nextstate_t.reshape([1,self.feature_size])
						state_t = state_t.reshape([1,self.feature_size])

						input_state[i] = state_t
						if(self.replay_mem.batch[i][4]==True):
							truth[i] = self.net.model.predict(state_t)
							truth[i][action_t] = reward_t
						else:
							q_target = reward_t + self.discount_factor*np.amax(self.prediction_net.model.predict(nextstate_t))
							truth[i] = self.net.model.predict(state_t)
							truth[i][action_t] = q_target
					
					if(curr_episode%self.print_loss_epi==0):
						self.net.model.fit(input_state,truth,epochs=1,verbose=1,batch_size = len(self.replay_mem.batch))
					else:
						self.net.model.fit(input_state,truth,epochs=1,verbose=0,batch_size = len(self.replay_mem.batch))

					nextstate = nextstate.reshape([1,self.feature_size])
					q_nextstate = self.net.model.predict(nextstate)
					nextaction = self.epsilon_greedy_policy(q_nextstate)
					curr_state = nextstate
					curr_action = nextaction

					iters += 1

					# if(iters%self.save_weights_iters==0):
					# 	self.net.save_model_weights(backup)

					if(iters%self.save_model_iters==0):
						if(self.replay==True):
							if(self.env_name == "CartPole-v0"):
								self.net.model.save('models/replay/cartpole/'+ str(iters) +'_cp_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'replay/cartpole'
							elif(self.env_name == "MountainCar-v0"):
								self.net.model.save('models/replay/mountaincar/'+ str(iters) +'_mc_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'replay/mountaincar'

						elif(self.deep==True):
							if(self.env_name == "CartPole-v0"):
								self.net.model.save('models/deep/cartpole/'+ str(iters) +'_cp_deep_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'deep/cartpole'
							elif(self.env_name == "MountainCar-v0"):
								self.net.model.save('models/deep/mountaincar/'+ str(iters) +'_mc_deep_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'deep/mountaincar'

						elif(self.duel==True):			
							if(self.env_name == "CartPole-v0"):
								self.net.model.save('models/duel/cartpole/'+ str(iters) +'_cp_duel_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'duel/cartpole'
							elif(self.env_name == "MountainCar-v0"):
								self.net.model.save('models/deep/mountaincar/'+ str(iters) +'_mc_duel_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
								folder = 'duel/mountaincar'

				self.epsilon -= self.epsilon_decay
				self.epsilon = max(self.epsilon, 0.05)
				
				# if(iters%self.update_prediction_net_iters==0):
				# 	self.prediction_net.load_model_weights(self.net.model.get_weights())
				# 	self.net.visualise_weights()
			###end of episode##

			self.prediction_net.load_model_weights(self.net.model.get_weights())

			max_reward = max(max_reward, curr_reward)

			if(len(reward_buf)>self.avg_rew_buf_size_epi):
				reward_buf.popleft()
			reward_buf.append(curr_reward)
			avg_reward = sum(reward_buf)/len(reward_buf)

			if(curr_episode%self.print_epi==0):
				print(curr_episode, curr_iters, self.epsilon ,int(avg_reward), int(curr_reward) ,complete)
			curr_episode += 1

			train_reward.append(curr_reward)

		final_weight_file = 'temp'
		for filename in os.listdir('models/'+folder):
			if(filename!='.DS_Store' and isfile('models/'+join(folder,filename))):
				r,_ = self.test('models/'+join(folder,filename),20)
				test_reward.append(r)
				final_weight_file = 'models/'+join(folder,filename)
		
		return train_reward, test_reward, final_weight_file

	def test(self, model_file, epi=100):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.net.load_model(model_file)
		reward_array = np.zeros(epi)

		tot_reward = 0
		for e in range(epi):	
			curr_reward = 0															
			curr_state = self.env.reset()
			curr_action = np.random.randint(self.action_size)
			while(True):
				nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
				curr_reward += reward
				if(is_terminal):
					break
				nextstate = nextstate.reshape([1,self.feature_size])
				q_nextstate = self.net.model.predict(nextstate)
				nextaction = self.greedy_policy(q_nextstate)
				
				curr_state = nextstate
				curr_action = nextaction

			tot_reward += curr_reward
			reward_array[e] = curr_reward

		self.evaluate = float(tot_reward/epi)

		return float(tot_reward/epi), reward_array

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		# Burn-in with random state and action transitions
		curr_mem_size = 0

		state = self.env.reset()
		action = np.random.randint(self.action_size)
		while(curr_mem_size<self.replay_mem.burn_in):
			nextstate, reward, is_terminal, _ = self.env.step(action)
			self.replay_mem.append([state,action,reward,nextstate,is_terminal])
			curr_mem_size += 1
			if(is_terminal == True):
				state = self.env.reset()
			else:
				state = nextstate
			action = np.random.randint(self.action_size)


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=bool,default=False)
	parser.add_argument('--train',dest='train',type=bool,default=True)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--deep',dest='deep',type=bool,default=False)
	parser.add_argument('--duel',dest='duel',type=bool,default=False)
	parser.add_argument('--replay',dest='replay',type=bool,default=False)
	return parser.parse_args()

def plot_graph(env_name, rewards, type_plot, algo):
	# plt.plot(range(0,len([3,4])),[3,4])
	# plt.show()
	plt.cla()
	plt.plot(range(0,len(rewards)),rewards)
	imgname = 'temp.png'
	if(type_plot == 'train'):
		imgname = 'train.png'
	elif(type_plot == 'test'):
		imgname = 'test.png'

	if(env_name=="CartPole-v0"):
		if('deep'==algo):
			plt.savefig('plots/deep/cartpole/'+imgname)
		if('duel'==algo):
			plt.savefig('plots/duel/cartpole/'+imgname)
		if('replay'==algo):
			plt.savefig('plots/replay/cartpole/'+imgname)
		else:
			plt.savefig('plots/linear/cartpole/'+imgname)
	elif(env_name=="MountainCar-v0"):
		if('deep'==algo):
			plt.savefig('plots/deep/mountaincar/'+imgname)
		if('duel'==algo):
			plt.savefig('plots/duel/mountaincar/'+imgname)
		if('replay'==algo):
			plt.savefig('plots/replay/mountaincar/'+imgname)
		else:
			plt.savefig('plots/linear/mountaincar/'+imgname)

def main(args):

	args = parse_arguments()
	environment_name = args.env
	env = gym.make(environment_name)
	
	#Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	agent = DQN_Agent(env,args.replay,args.deep,args.duel,args.render,args.env)
	train_reward,test_reward,final_weight_file = agent.train()

	# print(train_reward,test_reward)

	#plot discounted training and undiscounted testing rewards at intervals
	if(args.deep==True):
		plot_graph(args.env,train_reward,'train','deep')
		plot_graph(args.env,test_reward,'test','deep')
	if(args.duel==True):
		plot_graph(args.env,train_reward,'train','duel')
		plot_graph(args.env,test_reward,'test','duel')
	if(args.replay==True):
		plot_graph(args.env,train_reward,'train','replay')
		plot_graph(args.env,test_reward,'test','replay')
	else:
		plot_graph(args.env,train_reward,'train','linear')
		plot_graph(args.env,test_reward,'test','linear')

	#final test reward gets saves in .txt
	_, t = agent.test(final_weight_file)
	file_test = open('final_test.txt', 'w')

	if(args.deep==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-deep-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-deep-mean\t"+str(np.mean(t))+"\n")
		if(args.env=="CartPole-v0"):	
			file_test.write("cartpole-deep-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-deep-mean\t"+str(np.mean(t))+"\n")
	if(args.duel==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-duel-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-duel-mean\t"+str(np.mean(t))+"\n")
		if(args.env=="CartPole-v0"):	
			file_test.write("cartpole-duel-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-duel-mean\t"+str(np.mean(t))+"\n")
	if(args.replay==True):
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-replay-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-replay-mean\t"+str(np.mean(t))+"\n")
		if(args.env=="CartPole-v0"):	
			file_test.write("cartpole-replay-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-replay-mean\t"+str(np.mean(t))+"\n")
	else:
		if(args.env=="MountainCar-v0"):
			file_test.write("mountaincar-linear-std\t"+str(np.std(t))+"\n")
			file_test.write("mountaincar-linear-mean\t"+str(np.mean(t))+"\n")
		if(args.env=="CartPole-v0"):	
			file_test.write("cartpole-linear-std\t"+str(np.std(t))+"\n")
			file_test.write("cartpole-linear-mean\t"+str(np.mean(t))+"\n")


if __name__ == '__main__':
	main(sys.argv)
