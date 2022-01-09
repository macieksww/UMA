import csv
import numpy as np
import matplotlib.pyplot as plt

def read_result_file(file_name):
    with open(file_name) as result_file:
        csv_reader = csv.reader(result_file, delimiter=';')

        episodes = []
        rewards = []

        for row in csv_reader:
            episodes.append(float(row[0]))
            rewards.append(float(row[1]))
        return episodes, rewards

def plot_stats(file_name, r=False, avg=True, maximum=True, minimum=True):
    episodes, rewards = read_result_file(file_name)
    avg_r = []
    max_r = []
    min_r = []
    
    if r == True:
        plt.plot(episodes, rewards, label='rewards')
    if avg == True:
        # avg reward in the last 100 episodes
        for index in range(len(episodes)):
            avg_r.append(np.mean(rewards[0:index]))
            # if index == 1:
            #     avg_r.append(rewards[0])
            # elif index < 99 and index > 1:
            #     avg_r.append(np.mean(rewards[0:index]))
            # else:
            #     avg_r.append(np.mean(rewards[-100:]))
        plt.plot(episodes, avg_r, label='avg')
    if maximum == True:
        # max reward in the last 100 episodes
        for index in range(len(episodes)):
            index += 1
            max_r.append(max(rewards[0:index]))
            # if index == 1:
            #     max_r.append(rewards[0])
            # elif index < 99 and index > 1:
            #     max_r.append(np.max(rewards[0:index]))
            # else:
            #     max_r.append(np.max(rewards[-100:]))
        plt.plot(episodes, max_r, label='max')
    if minimum == True:
        # max reward in the last 100 episodes
        for index in range(len(episodes)):
            index += 1
            min_r.append(min(rewards[0:index]))
            # if index == 1:
            #     min_r.append(rewards[0])
            # elif index < 99 and index > 1:
            #     min_r.append(np.min(rewards[0:index]))
            # else:
            #     min_r.append(np.min(rewards[-100:]))
        plt.plot(episodes, min_r, label='min')
                
    
    
    
    plt.legend()
    plt.show()
            
        
plot_stats('sarsa_612612_ep_2000.csv', True, False)  
    
