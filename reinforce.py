import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
import math 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K

from keras.optimizers import Adam
import random
from tensorflow.python.ops import math_ops, clip_ops
from gym.wrappers import Monitor
import pickle

_EPSILON = 1e-7
def epsilon():                                                                                      # referenced from tensorflow sourcecode
    return _EPSILON

# def reinforce_loss(y_true, y_pred):                                                                 # referenced from saty
#     epsilon=tf.convert_to_tensor(_EPSILON,y_pred.dtype.base_dtype)

#     probable_acs=tf.clip_by_value(y_pred,epsilon,1-epsilon)
#     loss=tf.multiply(y_true,tf.log(probable_acs))#[0*P(0|S),0*P(1|S),G_t*P(2|S),0*P(3|S)]
#     loss_vector=K.sum(loss,axis=1) #0+0+G_t*P(2|S)+0
#     loss=tf.reduce_mean(loss_vector)#self explanatory
#     return -loss

def reinforce_loss(y_true, y_pred):                                                                 # referenced from tensorflow sourcecode
    y_pred = y_pred / math_ops.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
    # manual computation of crossentropy
    epsilon_ = tf.convert_to_tensor(epsilon(), y_pred.dtype.base_dtype)  
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)                                # clip so that not a log of zero                      
    return -4*math_ops.reduce_mean(y_true * math_ops.log(y_pred), axis=len(y_pred.get_shape()) - 1)

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        self.learning_rate = lr
        self.model.compile(optimizer = Adam(lr=self.learning_rate), loss=reinforce_loss)

        self.test_interval = 500
        self.save_model_interval = 500                                                                                             #reduced

    def run_model(self, env, type_run='train',render=False):
        # Generates an episode by running the cloned policy on the given env.
        return self.generate_episode(self.model, env, type_run, render)

    #dp af
    @staticmethod
    def G_t(rewards,gamma=1):       
        updated_rewards = [0]
        current_index = 1
        for t in range(len(rewards)-1,-1,-1):
            gt = rewards[t] + gamma*updated_rewards[current_index-1]
            # updated_rewards.append(gt/len(rewards))                                                                               #potential problem
            updated_rewards.append(gt)
            current_index +=1
        return list(reversed(updated_rewards[1:]))

    def scale_shit(self,tup):
        return tup[0]*tup[1]                                                                                            

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        std_list = []
        mean_list = []

        num_episodes = 50000

        save_episode_id=np.around(np.linspace(0,num_episodes,num=100))
        env = Monitor(env,'reinforce/videos/',video_callable= lambda episode_id: episode_id in save_episode_id, force=True)

        current_episode = 0
        while(current_episode<=num_episodes):
            #Generate episode as current training batch

            states,actions,rewards,num_steps = self.run_model(env)                                                              #actions are already one-hot
            current_batch_size = len(states)
            # print("Current Training Reward:",Reinforce.G_t(rewards,gamma)[0])
            actions = list(map(self.scale_shit,list(zip(actions,Reinforce.G_t(rewards,gamma)))))                                #potential problem
            history = self.model.fit(np.vstack(states),np.asarray(actions),epochs=1,verbose=0,batch_size=current_batch_size)
            loss = history.history['loss']
            # print(loss)

            if(current_episode%100==0):
                print("Episodes: {}, Loss: {}, Number of steps: {}".format(current_episode, loss, num_steps))
            if(current_episode%self.test_interval==0):
                # self.render_one_episode(env)
                std,mean = self.test(env,10)
                std_list.append(std)
                mean_list.append(mean)
                print(self.model.predict(states[len(states)-1]))
                print("Test Reward Std:{}, Test Mean Reward: {}".format(std,mean))
                with open('reinforce/trainreward_backup.pkl', 'wb') as f:
                                pickle.dump((std_list,mean_list), f)
            if(current_episode%self.save_model_interval==0):
                self.model.save("reinforce/"+"episode_"+str(current_episode))
            current_episode += 1

        self.model.save("reinforce/"+"episode_"+str(current_episode))
        return std_list, mean_list

    def render_one_episode(self,env):
        state = env.reset()
        action = np.random.randint(env.action_space.n)
        while(True):
            env.render()
            state = state.reshape([1,env.observation_space.shape[0]])
            nextstate, reward, is_terminal, _ = env.step(action)
            if(is_terminal == True):
                break                            
            state = nextstate.reshape([1,env.observation_space.shape[0]])
            action = np.random.choice(env.action_space.n,1,p=self.model.predict(state).flatten())[0]

    def make_one_hot(self, env, action):
        one_hot_action_vector = np.zeros(env.action_space.n)
        one_hot_action_vector[action] = 1
        return one_hot_action_vector

    def generate_episode(self, model, env, type_run='train',render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step

        states = []
        actions = []
        rewards = []
        downscale_factor = 0.01
        if(type_run=='test'):
            downscale_factor = 1
        
        num_steps = 0
        state = env.reset()
        action = np.random.choice(env.action_space.n,1,p=model.predict(state.reshape([1,env.observation_space.shape[0]])).flatten())[0]
        while(True):
            if(render==True):
                env.render()

            state = state.reshape([1,env.observation_space.shape[0]])
            states.append(state)                                                                            #storing reshaped state
            actions.append(self.make_one_hot(env,action))                                                   #storing one hot action target

            nextstate, reward, is_terminal, _ = env.step(action)
    
            rewards.append(reward*downscale_factor)                                                         #storing downscaled reward
            if(is_terminal == True):
                break                            
            state = nextstate.reshape([1,env.observation_space.shape[0]])
            action = np.random.choice(env.action_space.n,1,p=model.predict(state).flatten())[0]
            num_steps += 1 
        return states, actions, rewards, num_steps

    def test(self, env, num_episodes=100, render=False):
        current_episode = 0
        rewards = []
        while(current_episode<num_episodes):
            if(render==True):
                env.render()
            _,_,r,_ = self.run_model(env,'test',render)
            rewards.append(np.sum(r))
            current_episode +=1

        return np.std(rewards),np.mean(rewards)

def plot_af(data,save_file_name):
#input: data[0] = list(std), data[1] = list(mean)
    # x = np.arange(0,50100,500)
    y = np.asarray(data[1])
    x = np.arange(0,500*y.shape[0],500)
    e = np.asarray(data[0])
    # print(np.squeeze(x).shape, np.squeeze(y).shape, np.squeeze(e).shape)
    # print(y)

    plt.errorbar(np.squeeze(x), np.squeeze(y), np.squeeze(e), linestyle='None', marker='^')
    plt.savefig(str(save_file_name), bbox_inches='tight')
    plt.show()

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # Create the environment.
    env = gym.make('LunarLander-v2')
    
    ######## save and load as json later#########

    # Load the policy model from file. 
    # with open(model_config_path, 'r') as f:        
    #     model = keras.models.model_from_json(f.read())                                                        # if loading from json without weights
    model = keras.models.load_model(model_config_path,custom_objects={'reinforce_loss': reinforce_loss})        # if loading from my_saved_weights

    reinforce_agent = Reinforce(model, lr)
    reinforce_agent.train(env)

    with open('reinforce/trainreward_backup.pkl', 'r') as f:
        data = pickle.load(f)
    plot_af(data,'reinforce_train.png')


if __name__ == '__main__':
    main(sys.argv)
