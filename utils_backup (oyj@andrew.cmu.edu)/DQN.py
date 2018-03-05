#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
import collections
import time


class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env, replay, deep, duel):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.learning_rate = 0.001																								#HYPERPARAMETER1

		if(deep==False and duel==False): 
			print("Setting up linear network....")
			self.model = Sequential()
			self.model.add(Dense(env.action_space.n, input_dim = env.observation_space.shape[0], kernel_initializer='he_uniform'))
			self.model.add(Activation('linear'))
			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='Linear.png', show_shapes = True)
		
		elif(deep==True):	
			print("Setting up DDQN network....")
			self.model = Sequential()
			self.model.add(Dense(32, input_dim = env.observation_space.shape[0], kernel_initializer='he_uniform'))
			self.model.add(Activation('relu'))
			self.model.add(BatchNormalization())
			# self.model.add(Dropout(0.5))
			self.model.add(Dense(32, input_dim = 32, kernel_initializer='he_uniform'))
			self.model.add(Activation('relu'))
			self.model.add(BatchNormalization())
			# self.model.add(Dropout(0.5))
			self.model.add(Dense(32, input_dim = 32, kernel_initializer='he_uniform'))
			self.model.add(Activation('relu'))
			self.model.add(BatchNormalization())
			# self.model.add(Dropout(0.5))
			self.model.add(Dense(env.action_space.n, input_dim = 32, kernel_initializer='he_uniform'))
			self.model.add(Activation('linear'))
			print("Q-Network initialized.... :)\n")

			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='DDQN.png', show_shapes = True)
		
		elif(duel==True):			
			print("Setting up Dueling DDQN network....")
			inp = Input(shape=(env.observation_space.shape[0],))
			layer_shared1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(inp)
			layer_shared1 = BatchNormalization()(layer_shared1)
			layer_shared2 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared1)
			layer_shared2 = BatchNormalization()(layer_shared2)
			# layers_shared = layer_shared2(layer_shared1(inp))
			print("Shared layers initialized....")

			layer_v1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared2)
			# layer_v1 = BatchNormalization()(layer_v1)
			layer_a1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared2)
			# layer_a1 = BatchNormalization()(layer_a1)
			layer_v2 = Dense(1,activation='linear',kernel_initializer='he_uniform')(layer_v1)
			layer_a2 = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform')(layer_a1)
			# layer_v = layer_v2(layer_v1(layers_shared))
			# layer_a = layer_a2(layer_a1(layers_shared))
			print("Value and Advantage Layers initialised....")

			# layer_q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]), output_shape=(env.action_space.n,))([layer_v, layer_a])
			layer_q = Lambda(lambda x: x[0][:] + x[1][:] - K.mean(x[1][:]), output_shape=(env.action_space.n,))([layer_v2, layer_a2])

			print("Q-function layer initialized.... :)\n")

			self.model = Model(inp, layer_q)
			self.model.compile(optimizer = Adam(lr=self.learning_rate), loss='mse')
			plot_model(self.model, to_file='Dueling Double DQN.png', show_shapes = True)

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

	def __init__(self, env, replay, deep, duel, render):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.net = QNetwork(env,replay,deep,duel)
		self.prediction_net = QNetwork(env,replay,deep,duel)
		self.replay = replay
		self.deep = deep
		self.duel = duel
		self.env = env
		self.replay_mem = Replay_Memory(10000,500)																#HYPERPARAMETER2
		self.render = render
		self.feature_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n
		self.discount_factor = 1 	

		if(env == "CartPole-v0"):
			self.discount_factor = 0.99
		elif(env == "MountainCar-v0"):
			self.discount_factor = 1

		self.train_iters = 1000000
		self.epsilon = 0.5 																						#HYPERPARAMETER3
		self.epsilon_min = 0.05																					#HYPERPARAMETER4
		self.num_episodes = 4000
		self.epsilon_decay = float((self.epsilon-self.epsilon_min)/150000)										#HYPERPARAMETER5
		self.update_prediction_net_iters =500 																	#HYPERPARAMETER6
		self.avg_rew_buf_size_epi = 10 
		self.save_weights_iters = 5000 
		self.save_model_iters = 2000 															
		self.print_epi = 1 
		self.print_loss_epi = 50 

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
		curr_episode = 1
		iters = 1
		max_reward = 0
		reward_buf = collections.deque()

		self.burn_in_memory()

		# while(iters<self.train_iters): 																	#uncomment for cartpole
		for e in range(self.num_episodes):																	#uncomment for mountaincar
			curr_reward = 0
			curr_state = self.env.reset()
			curr_state = curr_state.reshape([1,self.feature_size])
			curr_action = self.epsilon_greedy_policy(self.net.model.predict(curr_state))
			
			# while(iters<self.train_iters): 																#uncomment for cartpole
			while(True): 																					#uncomment for mountaincar
				self.env.render()

				if(self.replay==False and self.deep==False and self.duel==False):

					nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
					curr_reward += reward
					# truth = np.zeros(shape=[1,self.action_size])

					if(is_terminal == True):
						q_target = reward
						truth = self.net.model.predict(curr_state)
						truth[0][curr_action] = q_target
						self.net.model.fit(curr_state,truth,epochs=1,verbose=0)
						break

					nextstate = nextstate.reshape([1,self.feature_size])
					q_nextstate = self.net.model.predict(nextstate)
					nextaction = self.epsilon_greedy_policy(q_nextstate)
					q_target = reward + self.discount_factor*np.amax(q_nextstate)

					truth = self.net.model.predict(curr_state)
					truth[0][curr_action] = q_target

					if(curr_episode%self.print_loss_epi==0):
						self.net.model.fit(curr_state,truth,epochs=1,verbose=1)
					else:
						self.net.model.fit(curr_state,truth,epochs=1,verbose=0)

					curr_state = nextstate
					curr_action = nextaction

					iters += 1

					# if(iters%self.save_weights_iters==0):
					# 	self.net.save_model_weights(backup)
					if(iters%self.save_model_iters==0):
						if(self.env == "CartPole-v0"):
							self.net.model.save('cp_BN_linear_nrp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
						elif(self.env == "MountainCar-v0"):
							self.net.model.save('mc_BN_linear_nrp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
				


				else:
					nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
					self.replay_mem.append([curr_state,curr_action,reward,nextstate,is_terminal])

					curr_reward += reward
					if(is_terminal):
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
							truth[i] = self.prediction_net.model.predict(state_t)
							truth[i][action_t] = reward_t
						else:
							q_target = reward_t + self.discount_factor*np.amax(self.prediction_net.model.predict(nextstate_t))
							truth[i] = self.prediction_net.model.predict(state_t)
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

					if(self.replay==True):
						if(self.env == "CartPole-v0"):
							self.net.model.save('cp_BN_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
						elif(self.env == "MountainCar-v0"):
							self.net.model.save('mc_BN_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')

					elif(self.deep==True):
						if(self.env == "CartPole-v0"):
							self.net.model.save('cp_BN_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
						elif(self.env == "MountainCar-v0"):
							self.net.model.save('mc_BN_linear_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')

					elif(self.duel==True):			
						if(self.env == "CartPole-v0"):
							self.net.model.save('cp_BN_duel_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')
						elif(self.env == "MountainCar-v0"):
							self.net.model.save('mc_BN_duel_rp_'+str(self.net.learning_rate)+'_'+str(self.replay_mem.burn_in)+'_'+str(self.replay_mem.memory_size)+'_'+'.h5')

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
				print(curr_episode, iters, self.epsilon ,avg_reward)
			curr_episode += 1

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory.
		self.net.load_model(model_file)
		
		curr_reward = 0
		for e in range(100):																
			nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
			curr_reward += reward
			if(is_terminal):
				break
			nextstate = nextstate.reshape([1,self.feature_size])
			q_nextstate = self.net.model.predict(nextstate)
			nextaction = self.epsilon_greedy_policy(q_nextstate)
			
			curr_state = nextstate
			curr_action = nextaction

		self.evaluate = float(curr_reward/100)

		pass

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		# Burn-in with random state and action transitions
		curr_mem_size = 0
		while(curr_mem_size<self.replay_mem.burn_in):
			state = self.env.reset()
			action = np.random.randint(self.action_size)
			while(curr_mem_size<self.replay_mem.burn_in):
				nextstate, reward, is_terminal, _ = self.env.step(action)
				if(is_terminal == True):
					break
				self.replay_mem.append([state,action,reward,nextstate,is_terminal])
				curr_mem_size += 1

				action = np.random.randint(self.action_size)
				state = nextstate


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


def main(args):


	args = parse_arguments()
	environment_name = args.env
	env = gym.make(environment_name)

	# Setting the session to allow growth, so it doesn't allocate all GPU memory.
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session.
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it.
	agent = DQN_Agent(env,args.replay,args.deep,args.duel,args.render)
	agent.train()
	agent.test()

	print(agent.evaluate)


if __name__ == '__main__':
	main(sys.argv)
