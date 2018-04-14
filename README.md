1) General:
-Videos for trained policies are in videos directory.
-imitation.py, reinforce.py and a2c.py are implementations for question 1,2 & 3 respectively.
-utils folder has some utility function used while solving the homework


2) Running the code:

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


