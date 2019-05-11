# Deep Reinforcement Learning with Open AI Gym  

This work was done for my final project in CSCI E-89 Deep Learning.  
It contains code to run a Deep Q Neural Network and extensions of DQNs on various Open AI Gym environments.  
After cloning run `pip install -r requirements.txt` to install all dependencies  
Run `python3 game.py` to try it out  

The supported agents at the moment are  
* dqn - simple deep Q-learning neural network  
* ddqn - double Q-learning network  
* duelingddqn - dueling architecture added to the double Q network  
* perddqn - prioritized experience replay added to the double Q network  
* test - use this to test out your own architecture or hyperparameters  
