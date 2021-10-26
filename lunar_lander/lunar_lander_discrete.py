# !usr/bin/

'''
Random agent - Only select random actions & track how well the agent performs

Goal: Teach an agent how to land a ship in the designated landing pad location at (0,0)
Episode ends when lander crashes or lands on surface
Fuel = infinite
Landing outside pad is possible

Rewards:
    + 200 for landing in pad,
    -0.3 for firing main engine / frame
    ...

Actions:
    0 - do nothing
    1 - left engine
    2 - down engine
    3 - right engine

Solution: Random

Packages:
pip install numpy
pip install gym
pip install box2d-py
pip install pyglet==1.5.11
conda install swig
'''


import numpy as np
import gym
from random import sample
import matplotlib.pyplot as plt

print('Numpy version: ', np.__version__)
print('Gym version: ', gym.__version__)

env = gym.make('LunarLander-v2')
# to get gym to run need to run "pip install box2d-py"
env.seed(0)

if __name__ == '__main__':
    print('Number of actions: ', env.action_space.n)
    print('Action space: ', env.action_space)
    print('Observation space: ', env.observation_space)

    n_episodes = 500

    total_reward_list = []
    total_timestep_list = []

    for episode in range(n_episodes):

        env.reset()
        episode_reward_list = []
        episode_timestep_list = []

        t = 1

        while True:

            action = env.action_space.sample()
            #action = sample(env.action_space.n, 1)

            env.render()
            next_state, reward, is_done, _ = env.step(action)

            episode_reward_list.append(reward)
            episode_timestep_list.append(t)

            t+=1

            if is_done:
                total_reward_list.append(episode_reward_list)
                total_timestep_list.append(episode_timestep_list)
                break

    sum_reward_list = [np.sum(i) for i in total_reward_list]
    plt.plot([i+1 for i in range(len(total_reward_list))], sum_reward_list)
    plt.title('Total rewards per episode')
    plt.show()
    plt.savefig('total_reward_random.png')

    avg_reward_list = [np.mean(i) for i in total_reward_list]
    plt.plot([i + 1 for i in range(len(total_reward_list))], avg_reward_list)
    plt.title('Average rewards per episode')
    plt.show()
    plt.savefig('avg_reward_random.png')







