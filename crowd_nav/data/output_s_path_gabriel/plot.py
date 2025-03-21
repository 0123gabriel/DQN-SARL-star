# -*- coding: utf-8 -*-

import re
from collections import defaultdict
import matplotlib.pyplot as plt

def calculate_weighted_average(values):
    total = sum(values)
    average = total / len(values)
    return average

file_path = 'output.log'

env_values_latest_file = defaultdict(lambda: defaultdict(list))

pattern = r' ([^:]+): ([-+]?\d+\.\d+)' 

with open(file_path, 'r') as file:
    current_env_latest_file = None
    file_iterator = iter(file)
    next(file_iterator) #First line

    for line in file:
        env_match_latest = re.search(r"running \d+/3000 episode, simulation environment: (\w+)", line)
        #print(env_match_latest)
        if env_match_latest:
            current_env_latest_file = env_match_latest.group(1)

        if current_env_latest_file:
            try:
                for match in re.finditer(pattern, next(file_iterator)):
                    key, value = match.groups()
                    if key in ['Success! Total Reward', 'Collision! Total Reward', 'Timeout! Total Reward', 'Collision Wall! Total Reward']:
                        key = 'Total_Reward'
                    env_values_latest_file[current_env_latest_file][key].append(float(value))

            except StopIteration:
                pass
        
        current_env_latest_file = None

#print(env_values_latest_file.items())

combined_averages_latest = {'Circle_crossing': defaultdict(list), 'Circle_static': defaultdict(list)}#, 'Static': defaultdict(list), 'No': defaultdict(list)}
dynamic_envs = ['circle_crossing']#, 'square_crossing']
mixed_envs = ['circle_static']#, 'square_static']
#static_envs = ['static']
#no_envs = ['no']

for env, metrics in env_values_latest_file.items():
    if env in dynamic_envs:
        category = 'Circle_crossing'
    elif env in mixed_envs:
        category = 'Circle_static'
    #elif env in static_envs:
    #    category = 'Static'
    #else:
    #    category = env.capitalize()

    for key, values in metrics.items():
        combined_averages_latest[category][key].extend(values)

print(combined_averages_latest.keys())

x_axis_values = [a for a in range(1, 1500)]
reward_list = []
reward_values = []  

for categ in combined_averages_latest.keys():
    for reward in combined_averages_latest[categ].keys():
        reward_list.append(reward)
        reward_list.append(reward)
        reward_values.append(combined_averages_latest[categ][reward][1:1500])
        reward_values.append(combined_averages_latest[categ][reward][1500:3000])

#print(len(reward_list))
#print(reward_list[0:len(reward_list)//2])

reward_list = reward_list[0:len(reward_list)//2]

final_averages_combined_latest = {
    category: {key: calculate_weighted_average(values) for key, values in metrics.items()}
    for category, metrics in combined_averages_latest.items()
}

#print(final_averages_combined_latest)
#print(len(combined_averages_latest['Circle_crossing']['Total_Reward'][0:1500]))

for i in range(len(reward_list)):
    x_axis_values = [a for a in range(len(reward_values[i]))]
    plt.figure()
    plt.plot(x_axis_values, reward_values[i])
    plt.title(reward_list[i] + (' imitation learning' if i%2 == 0 else ' test'))
plt.show()



