from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from collections import deque
import random
import numpy as np

# Agent to test different hyparameters and network architectures
class TestAgent:
    def __init__(self, game, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.training_frequency = 4
        self.target_update_frequency = 10000
        self.epsilon_decay_frequency = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.reset_target_model()
        self.csv_loss_logger = CSVLogger(game + '_test/' + game + '_test_loss.csv', append=True, separator=',')

    def _build_model(self):
        # Neural Net for a double Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def reset_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                # map state to future reward
                target_f = self.model.predict(state)
                target_f[0][action] = target
                # train model
                self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.csv_loss_logger])
        # update target model weights
        if total_step % self.target_update_frequency == 0:
            self.reset_target_model()
        if total_step % self.epsilon_decay_frequency == 0:
            # decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
