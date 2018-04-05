import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import sys 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#reinforce.py required in the same directory
from reinforce import reinforce_loss

model_path = str(sys.argv[1])

def render_one_episode(model,env,epi_num):
        rewards = []
        state = env.reset()
        state = state.reshape([1,env.observation_space.shape[0]])
        action = np.random.choice(env.action_space.n,1,p=model.predict(state).flatten())[0]
        while(True):
            env.render()
            state = state.reshape([1,env.observation_space.shape[0]])
            nextstate, reward, is_terminal, _ = env.step(action)
            rewards.append(reward)
            if(is_terminal == True):
                break                            
            state = nextstate.reshape([1,env.observation_space.shape[0]])
            action = np.random.choice(env.action_space.n,1,p=model.predict(state).flatten())[0]
        print("Total reward for {} : {}".format(epi_num,np.sum(rewards)))

def main():
	env = gym.make('LunarLander-v2')
	model = keras.models.load_model(model_path,custom_objects={'reinforce_loss': reinforce_loss})
	for i in range(10):
		render_one_episode(model,env,i)

if __name__ == '__main__':
	main()


