#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras import backend as K
import collections
import time
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization

class QNetwork():

	# This class essentially defines the network architecture.
	# The network should take in state of the world as an input,
	# and output Q values of the actions available to the agent as the output.

	def __init__(self, env):
		# Define your network architecture here. It is also a good idea to define any training operations
		# and optimizers here, initialize your variables, or alternately compile your model here.
		self.learning_rate = 0.0001
		print("Setting up network....")

		inp = Input(shape=(env.observation_space.shape[0],))
		layer_shared1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(inp)
		layer_shared1 = BatchNormalization()(layer_shared1)
		layer_shared2 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared1)
		layer_shared2 = BatchNormalization()(layer_shared2)
		# layers_shared = layer_shared2(layer_shared1(inp))
		print("Shared layers initialized....")

		layer_v1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared2)
		layer_v1 = BatchNormalization()(layer_v1)
		layer_a1 = Dense(32,activation='relu',kernel_initializer='he_uniform')(layer_shared2)
		layer_a1 = BatchNormalization()(layer_a1)
		layer_v2 = Dense(1,activation='linear',kernel_initializer='he_uniform')(layer_v1)
		# layer_v2 = BatchNormalization()(layer_v2)
		layer_a2 = Dense(env.action_space.n,activation='linear',kernel_initializer='he_uniform')(layer_a1)
		# layer_a2 = BatchNormalization()(layer_a2)
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
		print("\n")


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

	def __init__(self, env, render=False):

		# Create an instance of the network itself, as well as the memory.
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc.
		self.net = QNetwork(env)
		self.prediction_net = QNetwork(env)

		self.env = env
		self.replay_mem = Replay_Memory()
		self.render = render
		self.feature_size = env.observation_space.shape[0]
		self.action_size = env.action_space.n

		self.discount_factor = 1
		self.train_iters = 1000000
		self.epsilon = 0.5
		self.epsilon_min = 0.05
		self.num_episodes = 4000
		self.epsilon_decay = float((self.epsilon-self.epsilon_min)/100000)
		self.update_prediction_net_iters = 1000
		self.avg_rew_buf_size_epi = 10
		self.save_weights_iters = 5000
		self.save_model_iters = 2000
		self.print_epi = 1

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

		for e in range(self.num_episodes):
		# while(True):
			curr_reward = 0
			curr_state = self.env.reset()
			curr_state = curr_state.reshape([1,self.feature_size])
			curr_action = self.epsilon_greedy_policy(self.net.model.predict(curr_state))

			while(True):
			# while(iters<self.train_iters):
				# print(len(self.replay_mem.experience))
				self.env.render()
				nextstate, reward, is_terminal, debug_info = self.env.step(curr_action)
				self.replay_mem.append([curr_state,curr_action,reward,nextstate,is_terminal])

				curr_reward += reward
				if(is_terminal):
					break

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
					self.net.model.save('cartpole3x32_drop1_model_backup_deep_replay.h5')

				# print(self.epsilon, iters, curr_episode, curr_reward)
				self.epsilon -= self.epsilon_decay
				self.epsilon = max(self.epsilon, 0.05)
				if(iters%self.update_prediction_net_iters==0):
					self.net.visualise_weights()
				# 	self.prediction_net.load_model_weights(self.net.model.get_weights())
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
		pass

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.
		# Burn-in with random state and action transitions
		curr_mem_size = 0
		while(curr_mem_size<self.replay_mem.burn_in):
			state = self.env.reset()
			action = np.random.randint(self.action_size)
			nextstate, reward, is_terminal, _ = self.env.step(action)
			if(is_terminal==False):
				self.replay_mem.append([state,action,reward,nextstate,is_terminal])
				curr_mem_size += 1


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	# parser.add_argument('--replay',dest='replay',type=str,default=False)
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
	agent = DQN_Agent(env)
	agent.train()


if __name__ == '__main__':
	main(sys.argv)
