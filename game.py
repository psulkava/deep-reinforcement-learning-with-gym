import gym
import argparse
import sys
import os
import csv
import numpy as np
from collections import deque
from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent
from agents.duelingddqn import DuelingDDQNAgent
from agents.perddqn import PERDDQNAgent
from agents.test import TestAgent

class Game:

    def __init__(self):
        game, model, render, episode_limit, batch_size, target_score, test_model = self._args()
        self.env = gym.make(game)
        self.render = render
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.target_score = target_score
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.agent = DQNAgent(game, self.observation_space, self.action_space)
        self.save_name = str(game) + '_' + str(model.lower()) + '/' + str(game) + '_' + str(model.lower())
        if model.lower() == 'dqn':
            self.agent = DQNAgent(game, self.observation_space, self.action_space)
        elif model.lower() == 'ddqn':
            self.agent = DDQNAgent(game, self.observation_space, self.action_space)
        elif model.lower() == 'duelingddqn':
            self.agent = DuelingDDQNAgent(game, self.observation_space, self.action_space)
        elif model.lower() == 'perddqn':
            self.agent = PERDDQNAgent(game, self.observation_space, self.action_space)
        elif model.lower() == 'test':
            self.agent = TestAgent(game, self.observation_space, self.action_space)
        self.history = [('episode', 'score', 'average_score', 'steps', 'total_steps')]
        if test_model:
            self.agent.load_model(test_model)
            self.test()
        else:
            # make a directory to hold the saved files from this run if it doesn't exist
            try:
                os.mkdir(str(game) + '_' + str(model.lower()))
            except FileExistsError:
                pass
            self.train()

    def train(self):
        try:
            score_history = deque(maxlen=100)
            total_steps = 0
            for episode in range(self.episode_limit):
                # reset state at beginning of game
                state = self.env.reset()
                state = np.reshape(state, [1, self.observation_space])
                step = 0
                score = 0
                while True:
                    # increment step for each frame
                    step += 1
                    total_steps += 1

                    # render game
                    if self.render:
                        self.env.render()

                    # decide what action to take
                    action = self.agent.act(state)

                    # advance game to next frame based on the chosen action
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.observation_space])

                    # adjust score
                    score += reward

                    # add to memory
                    self.agent.remember(state, action, reward, next_state, done)

                    # train agent with experience
                    self.agent.replay(total_steps, self.batch_size)

                    if done:
                        score_history.append(score)
                        average_score = np.mean(score_history)
                        self.history.append((episode, score, average_score, step, total_steps))
                        # print episode, score, steps, and total steps and break loop
                        print("Episode: {}/{}, Score: {}, Average Score: {:.2f}, Steps: {}, Total Steps: {}".format(episode, self.episode_limit, score, average_score, step, total_steps))
                        # check if goal has been met
                        if self.target_score and average_score >= self.target_score:
                            print('Target score reached after {} episodes and {} total steps!'.format(episode, total_steps))
                            filename = self.save_name + '_final.h5'
                            print('Saving final model to ' + filename)
                            self.agent.save_model(filename)
                            self.exit()
                        break

                    # if not done make the next state the current state for the next iteration
                    state = next_state

                # save model every 500 episodes
                if episode % 100 == 0:
                    filename = self.save_name + '_' + str(episode) + '.h5'
                    print('Saving model to ' + filename)
                    self.agent.save_model(filename)
            self.exit()
        except KeyboardInterrupt:
            # Catch ctrl-c and gracefully end game
            filename = self.save_name + '_final.h5'
            print('Saving model to ' + filename)
            self.agent.save_model(filename)
            self.exit()
        except:
            self.env.close()
            sys.exit()

    def test(self):
        try:
            score_history = deque(maxlen=100)
            total_steps = 0
            for episode in range(100):
                # reset state at beginning of game
                state = self.env.reset()
                state = np.reshape(state, [1, self.observation_space])
                step = 0
                score = 0
                while True:
                    # increment step for each frame
                    step += 1
                    total_steps += 1

                    # render game
                    if self.render:
                        self.env.render()

                    # decide what action to take
                    action = self.agent.test_act(state)

                    # advance game to next frame based on the chosen action
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.observation_space])

                    # adjust score
                    score += reward

                    if done:
                        score_history.append(score)
                        average_score = np.mean(score_history)
                        self.history.append((episode, score, average_score, step, total_steps))
                        # print episode, score, steps, and total steps and break loop
                        print("Episode: {}/99, Score: {}, Average Score: {:.2f}, Steps: {}, Total Steps: {}".format(episode, score, average_score, step, total_steps))
                        break

                    # if not done make the next state the current state for the next iteration
                    state = next_state
            self.env.close()

        except:
            # Catch ctrl-c and gracefully end game
            print('Killing game')
            self.env.close()
            sys.exit()

    def exit(self):
        filename = self.save_name + '_history.csv'
        print('Saving training history to csv ' + filename)
        with open(filename, 'w') as out:
            csv_out = csv.writer(out)
            for row in self.history:
                csv_out.writerow(row)
        print('Killing game')
        self.env.close()
        sys.exit()

    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--game', help='Name of the Open AI Gym game to play', default='CartPole-v1')
        parser.add_argument('-m', '--model', help='Name of the model agent to use', default='dqn')
        parser.add_argument('-r', '--render', help='Whether to render the game or not', default=False, type=bool)
        parser.add_argument('-el', '--episode_limit', help='Number of episodes to run', default=5000, type=int)
        parser.add_argument('-bs', '--batch_size', help='Batch size to use', default=32, type=int)
        parser.add_argument('-ts', '--target_score', help='Target average score over last 100 episodes to stop after reaching', default=None, type=int)
        parser.add_argument('-tm', '--test_model', help='Filename of model weights to test performance of', default=None)
        args = parser.parse_args()
        game = args.game
        model = args.model
        render = args.render
        episode_limit = args.episode_limit
        batch_size = args.batch_size
        target_score = args.target_score
        test_model = args.test_model
        return game, model, render, episode_limit, batch_size, target_score, test_model

if __name__ == '__main__':
    Game()
