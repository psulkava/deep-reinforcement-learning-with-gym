from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from collections import deque
import random
import numpy as np

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, game, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.training_frequency = 4
        self.model = self._build_model()
        self.csv_loss_logger = CSVLogger(game + '_dqn/' + game + '_dqn_loss.csv', append=True, separator=',')

    def _build_model(self):
        # Neural Net for a simple Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # take random action
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def test_act(self, state):
        '''
        Don't take random actions when testing agent
        '''
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, total_step, batch_size):
        if len(self.memory) < batch_size:
            return
        if total_step % self.training_frequency == 0:
            # take random sample of events of size batch_size from memory
            minibatch = random.sample(self.memory, batch_size)
            # calculate q value for each event and train model
            for state, action, reward, next_state, done in minibatch:
                # target is current reward
                target = reward
                if not done:
                    # predict future reward
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                # map state to future reward
                target_f = self.model.predict(state)
                target_f[0][action] = target
                # train model
                self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.csv_loss_logger])
        # decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
