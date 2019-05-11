from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Multiply
from keras.layers import Add
from keras.layers import concatenate
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
import keras.backend as K
from collections import deque
import random
import numpy as np
from agents.SumTree import SumTree

# Double Deep Q-learning Agent with Prioritized Experience Replay
class PERDDQNAgent:
    def __init__(self, game, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(1000000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.training_frequency = 4
        self.target_update_frequency = 10000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.reset_target_model()
        self.csv_loss_logger = CSVLogger(game + '_perddqn/' + game + '_perddqn_loss.csv', append=True, separator=',')

    def _build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_size, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        ''' Can uncomment this to try out the dueling architecture with PER
        state_input = Input(shape=(self.state_size,))
        dense1 = Dense(24, activation='relu')(state_input)
        hidden1 = Dense(48, activation='relu')(dense1)
        q_prediction = Dense(self.action_size)(hidden1)
        hidden2 = Dense(48, activation='relu')(dense1)
        state_prediction = Dense(1)(hidden2)
        # Q = State value + (Action value - Average of all action values)
        q_prediction = Lambda(lambda x: x-K.mean(x, axis=-1), output_shape=(self.action_size,))(q_prediction)
        state_prediction = Lambda(lambda state_prediction: K.tile(state_prediction, [1, self.action_size]))(state_prediction)
        target_q = Add()([state_prediction, q_prediction])

        model = Model(inputs=state_input, outputs=target_q)
        '''
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def reset_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)

    def remember(self, state, action, reward, next_state, done):
        sample = (state, action, reward, next_state, done)
        state, target_f, error = self.getTargets(sample)
        self.memory.add(error, sample)

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

    def getTargets(self, sample):
        state, action, reward, next_state, done = sample
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
        target_f = self.model.predict(state)
        error = abs(target_f[0][action] - target)
        target_f[0][action] = target
        return (state, target_f, error)

    def replay(self, total_step, batch_size):
        if len(self.memory) < batch_size:
            return
        if total_step % self.training_frequency == 0:
            # take random sample of events of size batch_size from memory
            minibatch = self.memory.sample(batch_size)
            # calculate q value for each event and train model
            for (idx, sample) in minibatch:
                state, target_f, error = self.getTargets(sample)
                self.memory.update(idx, error)
                self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.csv_loss_logger])
        # update target model weights
        if total_step % self.target_update_frequency == 0:
            self.reset_target_model()
        # decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)


class Memory:   # stored as ( state, action, reward, next_state, done ) in SumTree
    epsilon = 0.01
    alpha = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def __len__(self):
        return len(self.tree)

    def _getPriority(self, error):
        return (error + self.epsilon) ** self.alpha

    def add(self, error, sample):
        priority = self._getPriority(error)
        self.tree.add(priority, sample)

    def sample(self, sample_size):
        batch = []
        segment = self.tree.total() / sample_size

        for i in range(sample_size):
            start = segment * i
            end = segment * (i + 1)

            sample = random.uniform(start, end)
            (idx, priority, data) = self.tree.get(sample)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        priority = self._getPriority(error)
        self.tree.update(idx, priority)
