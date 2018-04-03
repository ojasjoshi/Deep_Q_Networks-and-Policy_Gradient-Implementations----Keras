import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K

from keras.layers import Dense, Activation, Dropout, Input, Lambda, Add, Subtract
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.optimizers import Adam
import random
from gym.wrappers import Monitor
import operator
import math
import pickle

#reinforce.py required in the same directory
from reinforce import Reinforce
from reinforce import reinforce_loss as actor_loss
from reinforce import plot_af

# reference for a badass critic loss
def critic_loss(y_true, y_pred):                                                                                                                     # referenced from tensorflow sourcecode
  return K.mean(K.square(y_pred - y_true), axis=-1)

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic. (this class inherits the Reinforce class)

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        super(A2C,self).__init__(model,lr)                                                                                                            # hmmmm
        self.model = model
        self.critic_model = critic_model                                                                                                              # same as actor model for now (with (1,) output)
        self.learning_rate = lr
        self.critic_learning_rate = critic_lr
        self.n = n
  
        self.model.compile(optimizer = Adam(lr=self.learning_rate), loss=actor_loss, metrics=['acc'])
        self.critic_model.compile(optimizer = Adam(lr=self.critic_learning_rate), loss=critic_loss, metrics=['acc'])                                        # 'mse' **hazardous** potential problem because batch doesnt take average


    def V_t(self,states):
        val_preds = [self.critic_model.predict(states[t])[0] for t in range(len(states))]
        return val_preds

    # not dp af for large values of self.n
    def R_t_util(self, rewards, t, T, gamma=1.0):
        ret = 0
        for k in range(self.n):
            if(t+k<T):
                ret += pow(gamma,k)*rewards[t+k]
        return ret

    def R_t(self, rewards, value_preds, gamma=1.0):                                                                                                    # not static because need 'n'
        updated_rewards = []
        T = len(rewards)
        N = self.n
        for t in range(T-1,-1,-1):
            V_end = 0
            if(N+t<T):
                V_end = value_preds[t+N]
            rt = pow(gamma,N)*V_end + self.R_t_util(rewards,t,T,gamma)
            updated_rewards.append(rt)
        return list(reversed(updated_rewards))

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        std_list = []
        mean_list = []

        num_episodes = 50000

        save_episode_id=np.around(np.linspace(0,num_episodes,num=100))
        env = Monitor(env,'A2C/videos/',video_callable= lambda episode_id: episode_id in save_episode_id, force=True)

        current_episode = 0
        while(current_episode<=num_episodes):
            #Generate episode as current training batch
            states,actions,rewards,num_steps = super(A2C,self).run_model(env)                                                                         # actions are already one-hot
            current_batch_size = len(states)

            value_predictions = self.V_t(states)
            updated_rewards = self.R_t(rewards,value_predictions,gamma)

            ## printing all the shit in the world        
            # print(np.asarray(value_predictions).shape)
            # print(np.asarray(updated_rewards).shape)
            # print(np.asarray(list(map(operator.sub, updated_rewards, value_predictions))).shape)

            #backprop actor model
            actions = list(map(super(A2C,self).scale_shit,list(zip(actions,list(map(operator.sub, updated_rewards, value_predictions))))))           # potential **hazardous** potential problem
            history_actor = self.model.fit(np.vstack(states),np.asarray(actions),epochs=1,verbose=0,batch_size=current_batch_size)                   # check for gradient leak in critic network    
            acc_actor = history_actor.history['acc']
            loss_actor = history_actor.history['loss']

            #backprop critique model
            history_critic = self.critic_model.fit(np.vstack(states),np.asarray(updated_rewards),epochs=1,verbose=0,batch_size=current_batch_size)
            acc_critic = history_critic.history['acc']
            loss_critic = history_critic.history['loss']

            if(current_episode%100==0):
                print("Episodes: {}, Actor_Loss: {}, Critic_Loss:{}, Number of steps: {}".format(current_episode, loss_actor, loss_critic, num_steps))
            if(current_episode%self.test_interval==0):
                # self.render_one_episode(env)
                std,mean = super(A2C,self).test(env)
                std_list.append(std)
                mean_list.append(mean)
                print(self.model.predict(states[len(states)-1]))
                print("Test Reward Std:{}, Test Mean Reward: {}".format(std,mean))
                with open('A2C/trainreward_backup.pkl', 'wb') as f:
                                pickle.dump((std_list,mean_list), f)
            if(current_episode%self.save_model_interval==0):
                self.model.save("A2C/actor/"+"episode_"+str(current_episode))
                self.critic_model.save("A2C/critic/"+"episode_"+str(current_episode))
            current_episode += 1

        self.model.save("A2C/actor/"+"episode_"+str(current_episode))
        self.critic_model.save("A2C/critic/"+"episode_"+str(current_episode))
        return std_list, mean_list


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path_actor', dest='model_config_path_actor',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--model-config-path_critic', dest='model_config_path_critic',
                        type=str, default='',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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
    model_config_path_actor = args.model_config_path_actor
    model_config_path_critic = args.model_config_path_critic
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    with open(model_config_path_actor, 'r') as f:
        model = keras.models.model_from_json(f.read())
    # plot_model(model, to_file='actor_model.png', show_shapes = True)           

    # Critic Model
    inp = Input(shape=(env.observation_space.shape[0],))
    layer_dense = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(inp)
    layer_dense = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_dense)
    layer_dense = Dense(16,activation='relu',kernel_initializer='he_uniform',use_bias = True)(layer_dense)
    layer_v = Dense(1,activation='linear',kernel_initializer='he_uniform',use_bias = True)(layer_dense)

    critic_model = Model(inp, layer_v)
    # plot_model(critic_model, to_file='critic_model.png', show_shapes = True)           
    # critic_model.summary()

    ## loading saved models
    # model = keras.models.load_model(model_config_path_actor,custom_objects={'actor_loss': actor_loss})        # if loading from my_saved_weights
    # critic_model = keras.models.load_model(model_config_path_critic,custom_objects={'actor_loss': actor_loss})                                                   # if loading from my_saved_weights

    A2C_agent = A2C(model,lr,critic_model,critic_lr,n)
    A2C_agent.train(env)

    # plot training 
    with open('A2C/trainreward_backup.pkl', 'r') as f:
        data = pickle.load(f)
    plot_af(data,'A2C_train.png')

if __name__ == '__main__':
    main(sys.argv)
