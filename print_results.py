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
            # avg reward till ceratian episod
            avg_r.append(np.mean(rewards[0:index]))
            
            # # avg reward in the last 100 episodes
            # if index < 99:
            #     avg_r.append(np.mean(rewards[0:index]))
            # else:
            #     avg_r.append(np.mean(rewards[-100:]))
        plt.plot(episodes, avg_r, label='avg')
    if maximum == True:
        for index in range(len(episodes)):
            index += 1
            # max reward till ceratian episod
            max_r.append(max(rewards[0:index]))
            
            # # min reward in the last 100 episodes
            # if index < 100:
            #     max_r.append(np.max(rewards[0:index]))
            # else:
            #     max_r.append(np.max(rewards[-100:]))
        plt.plot(episodes, max_r, label='max')
    if minimum == True:
        for index in range(len(episodes)):
            index += 1
            # max reward till ceratian episod
            min_r.append(min(rewards[0:index]))
            
            # # max reward in the last 100 episodes
            # if index < 100:
            #     min_r.append(np.min(rewards[0:index]))
            # else:
            #     min_r.append(np.min(rewards[-100:]))
        plt.plot(episodes, min_r, label='min')
                
    plt.legend()
    plt.show()
            
        
plot_stats('sarsa_612612_ep_2000.csv', True)  
    
