from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import Multiply
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import numpy as np
import gym
from collections import deque
import random
from atari_wrappers import *
import matplotlib.pyplot as plt

# Deep Q-learning Agent
class DDQNAgent:
    def __init__(self, game, state_size, action_size):
        self.game = game
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=700000)
        self.gamma = 0.99    # discount rate
        self.rho = 0.95
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.000001059
        self.learning_rate = 0.00025
        self.training_step = 4
        self.target_update_frequency = 40000
        self.save_frequency = 10000
        self.replay_start_size = 50000
        self.model = self._build_model()
        #self.target_model = self._build_model()
        #self.reset_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(32, 8, input_shape=self.state_size, strides=(4, 4), activation='relu'))
        model.add(Convolution2D(64, 4, input_shape=self.state_size, strides=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, input_shape=self.state_size, strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        optimizer = RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=0.01)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def save_model(self, step):
        print('Saving model: atari' + str(self.game) + str(step) + '.h5')
        self.model.save('atari' + str(self.game) + str(step) + '.h5')

    #def reset_target_model(self):
        #self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon or len(self.memory) < self.replay_start_size:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(np.asarray(state), axis=0))
        return np.argmax(act_values[0])  # returns action

    def replay(self, step, batch_size):
        if len(self.memory) < self.replay_start_size:
            return
        if step % self.training_step == 0:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                state = np.expand_dims(np.asarray(state), axis=0)
                next_state = np.expand_dims(np.asarray(next_state), axis=0)
                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
        #if step % self.target_update_frequency == 0:
            #self.reset_target_model()
        if step % self.save_frequency == 0:
            self.save_model(step)

# initialize gym environment and the agent
game = 'Pong'
env = wrap_deepmind(gym.make(game + 'Deterministic-v4'), frame_stack=True)
observation_space = env.observation_space.shape
action_space = env.action_space.n
print('Observation space: ', observation_space)
print('Action space: ', action_space)
agent = DDQNAgent(game, observation_space, action_space)
episodes = 100000
total_step = 0
# Iterate the game
for e in range(episodes):
    # reset state in the beginning of each game
    state = env.reset()
    # time_t represents each frame of the game
    # Our goal is to keep the pole upright as long as possible until score of 500
    # the more time_t the more score
    step = 0
    score = 0
    while True:
        total_step += 1
        step += 1
        # turn this on if you want to render
        #env.render()
        # Decide action
        action = agent.act(state)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, done, _ = env.step(action)
        score += reward
        # Remember the previous state, action, reward, and done
        agent.remember(state, action, reward, next_state, done)
        # make next_state the new current state for the next frame.
        state = next_state
        # train the agent with the experience of the episode
        agent.replay(total_step, 32)
        # done becomes True when the game ends
        # ex) The agent drops the pole
        if done:
            # print the score and break out of the loop
            print("episode: {}/{}, steps: {}, total steps: {}, score: {}"
                  .format(e, episodes, step, total_step, score))
            break

