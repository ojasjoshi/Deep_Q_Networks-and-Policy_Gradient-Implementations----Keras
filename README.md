Code Implementation by Ojas Joshi(oyj), Ayush Raina (araina)

The folder contains 2 functions.



1. DQN.py
The function contains 6 important parameters that can be tweaked according the model requirements.
Following arguments are required while calling the function:

a) Linear QNetwork: 
> python3 DQN.py --env "environment_name" 

b) Linear Replay QNetwork:
> python3 DQN.py --env "environment_name" --replay True 

c) Deep QNetwork:
> python3 DQN.py --env "environment_name" --deep True 

d) Duel QNetwork:
> python3 DQN.py --env "environment_name" --duel True

For rendering the environment, --render True can be used. In order to save the weights,videos and plots during training, following directory structure must be used for the code to work. In addition, save_w on line 495 should be set to True.

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


(optional)
2. DQN_plot.py
After running DQN.py, run >python3 DQN_plot.py with the exact arguments to plot the training and testing plots and the mean and standard deviation of the final test run. The results of the final test run are stored in test_final.txt in the main directory.









