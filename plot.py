import matplotlib.pyplot as plt
import pandas as pd

def plot_game_results(loss_file, history_file):
    loss = pd.read_csv(loss_file)
    plt.plot(loss['loss'])
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.show()
    history = pd.read_csv(history_file)
    plt.plot(history['score'], label='Score')
    plt.plot(history['average_score'], label='Average Score')
    plt.legend(loc='upper left')
    plt.xlabel('Episode #')
    plt.show()
