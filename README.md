Updated 04.14.2018

Code Implementation by Ojas Joshi(oyj)

--- DQN/Dueling network(MountainCar-v0/CartPole-v0)

DQN.py
The function contains 6 important parameters that can be tweaked according the model requirements.

TRAINING
Following arguments are required while calling the function:

a) Linear QNetwork: 
> python3 DQN.py --env "environment_name" 

b) Linear Replay QNetwork:
> python3 DQN.py --env "environment_name" --replay True 

c) Deep QNetwork:
> python3 DQN.py --env "environment_name" --deep True 

d) Duel QNetwork:
> python3 DQN.py --env "environment_name" --duel True

For rendering the environment, --render True can be used. A backup model is saved every 10000 iterations in the main directory.
In order to save the weights,videos and plots during training, following directory structure must be used for the code to work. In addition, save_w on line 495 should be set to True.

main_dir
|	models
		|deep
			|cartpole,mountaincar
		|duel
			|cartpole,mountaincar
		|linear
			|cartpole,mountaincar
		|replay
			|cartpole,mountaincar
|	plots
		|deep
			|cartpole,mountaincar
		|duel
			|cartpole,mountaincar
		|linear
			|cartpole,mountaincar
		|replay
			|cartpole,mountaincar
| 	graphs
|	data
|	videos	

TESTING

>python3 DQN.py --env "environment_name" --deep/--duel/--replay True --model "<path-to-model-file>" --train False

-- DQN_plot.py
After running DQN.py, run >python3 DQN_plot.py with the exact arguments to plot the training and testing plots and the mean and standard deviation of the final test run. The results of the final test run are stored in test_final.txt in the main directory.


---- Policy Gradient (LunarLander-v0): 

-Before training:

a) reinforce - create a folder named 'reinforce' in the same directory
b) a2c - create a folder 'A2C' in the same directory. Make two subfolders 'actor' and 'critic' in A2C folder.
(make sure reinforce.py is in the same directory as a2c.py. If not make changes in importing the super class and reinforce loss function in a2c.py)

-Training:
--To run training from scratch:

> python imitation.py

> python reinforce.py

> python a2c.py --n (value of N)

--To load a model before training: 

(uncomment line 246 and comment line 245)
> python reinforce.py --model-config-path <path-to-model>

(uncomment lines 206,207 and comment lines 188,189)
> python a2c.py --model-config-path_actor <path-to-actor-model> --model-config-path_critic <path-to-critic-model> --n (value of N)

-Testing trained models: Uncomment lines 252,253 reinforce.py and lines 213-215 for a2c.py


