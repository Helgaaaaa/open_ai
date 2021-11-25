'''
Train agent to learn how to land on a landing pad at (0,0)

TODO: Parameters/Experiments:
1. Experience replay buffer size (100k, 1M, 10M) - DONE
2. Learning rate (alpha = 0.01, 0.005, 0.001) - ONGOING
3. Epsilon min & decay rate
    (epsilon min = 0.01, 0.05, 0.1)
    (epsilon decay = 0.9, 0.95, 0.99)
4. Discount rate (gamma = 0.9, 0.95, 0.99) - DONE
5. neural network architecture (hidden layers)
6. custom reward function
7. Function approximation methods:
    1. value-based
    2. policy-based**
    3. CNN
8. Batch size (32 vs. 64)
9.
'''

import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import time
from datetime import datetime

env = gym.make('LunarLander-v2')
env.seed(0)

class KerasDQN:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = deque(maxlen=1000000)

        # agent parameters
        self.epsilon = 1
        self.epsilon_decay = 0.996 # how quickly to decay exploration vs. exploitation every episode
        self.epsilon_min = 0.1 # always explore at least 1% of the time
        self.gamma = 0.99 # how much you care about immediate vs. future rewards

        # nn parameters
        self.batch_size = 64
        self.alpha = 0.001
        self.model = self.build_model()

    def build_model(self):

        model = keras.Sequential(
            [
                keras.Input(shape=(self.state_space)),
                layers.Dense(150, activation='relu', name='first_layer'), # input_size = state_size
                layers.Dense(120, activation='relu', name='second_layer'),
                layers.Dense(self.action_space, activation='linear', name='output_layer') # output_size = action_size
            ]
        )
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha)) # TODO: Do we want to try other optimizers
        return model

    def remember(self, state, action, reward, is_done, next_state):
        self.memory.append((state, action, reward, is_done, next_state))

    def act(self, state):
        if np.random.uniform(0,1) <= self.epsilon:
            action = random.randrange(self.action_space) # choose random action
        else:
            action = np.argmax(self.model.predict(state)) # choose greedy action
        return action

    def replay(self):

        if len(self.memory) < self.batch_size:
            return None # should I use break or continue or pass here

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.squeeze(np.array([i[0] for i in minibatch]))
        action_batch = np.array([i[1] for i in minibatch])
        reward_batch = np.array([i[2] for i in minibatch])
        is_done_batch = np.array([i[3] for i in minibatch])
        next_state_batch = np.squeeze(np.array([i[4] for i in minibatch]))

        # calculate a target q-value we are using to calculate the loss
        # Bellman equation = reward + discounted future rewards of next_state
        q_target = reward_batch +\
                    self.gamma *\
                    (np.amax(self.model.predict_on_batch(next_state_batch), axis=1)) *\
                    (1 - is_done_batch)

        q_policy = self.model.predict_on_batch(state_batch)
        q_policy[[i for i in range(self.batch_size)], [action_batch]] = q_target
        self.model.fit(state_batch, q_policy, epochs=1, verbose=0)

        # decay the exploration until min - but not sure want to do this each time we update
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':

    # store the experiments
    experiment_details = 'learning_rate'
    experiment_total_reward_results = []
    experiment_avg_reward_results = []
    values_to_test = [0.01, 0.005, 0.001]

    for value in values_to_test:

        start = time.time()
        time_now = datetime.now().strftime('%m_%d_%Y_%H_%M')

        n_episodes = 250
        max_frames_per_episode = 500
        agent = KerasDQN(env.action_space.n, env.observation_space.shape[0])
        agent.alpha = value

        experiment_info = 'Experiment Date: ' + str(time_now) +\
                          ' | Episodes= ' + str(n_episodes) +\
                          ' | max_frames= ' + str(max_frames_per_episode) +\
                          ' | learning_rate= ' + str(agent.alpha) +\
                          ' | batch_size= ' + str(agent.batch_size) +\
                          ' | gamma= ' + str(agent.gamma) +\
                          ' | epsilon_decay= ' + str(agent.epsilon_decay) +\
                          ' | epsilon_min= ' + str(agent.epsilon_min) +\
                          ' | memory_size= ' + str(agent.memory) +\
                          ' | experiment_run= ' + str(experiment_details)

        print('------------------ Starting Lunar Lander Training ------------------ ')
        print('Number of actions: ', env.action_space.n)
        print('Observation space: ', env.observation_space.shape[0])
        print('\n', experiment_info)
        print('-------------------------------------------------------------------- ')

        all_episode_total_rewards = []          # list containing total scores per episode
        all_episode_avg_rewards = []            # list containing average reward over last 50 episodes
        reward_window = deque(maxlen=25)        # total scores in most recent 100 episodes

        for episode in range(n_episodes):

            episode_reward = 0
            num_frames = 1

            state = env.reset()
            state = state.reshape((1, -1))  # (1,8)

            for frame in range(max_frames_per_episode):

                action = agent.act(state)
                env.render()
                next_state, reward, is_done, _ = env.step(action)
                agent.remember(state, action, reward, is_done, next_state)
                episode_reward += reward

                if frame == (max_frames_per_episode - 1):
                    is_done = True
                # elif np.mean(reward_window) >= 200: # if solved
                #     is_done = True

                if is_done:
                    print(f'Complete episode: {episode} | Total reward: {episode_reward} | Total frames: {num_frames}')
                    all_episode_total_rewards.append(episode_reward)
                    reward_window.append(episode_reward)
                    all_episode_avg_rewards.append(np.mean(reward_window))
                    break

                num_frames += 1

                state = next_state
                state = state.reshape((1, -1))

                agent.replay()

        experiment_total_reward_results.append(all_episode_total_rewards)
        experiment_avg_reward_results.append(all_episode_avg_rewards)

        print(f'Ran {n_episodes} in ', round((time.time() - start) / 60, 2), ' minutes.')

    # Plot after running experiment:
    x = [i + 1 for i in range(len(all_episode_total_rewards))]
    for i in range(len(values_to_test)):
        sns.lineplot(x, experiment_total_reward_results[i], label=str(values_to_test[i]))
    plt.title('Total rewards per episode | Experiment ' + str(experiment_details) + ' | Runtime ' + str(time_now))
    plt.show()
    plt.savefig(f'experiment_results/total_reward_dqn_{time_now}.png')

    x = [i + 1 for i in range(len(all_episode_avg_rewards))]
    for i in range(len(values_to_test)):
        sns.lineplot(x, experiment_avg_reward_results[i], label=str(values_to_test[i]))
    plt.title('Average rewards over 25 episodes | Experiment ' + str(experiment_details) + ' | Runtime ' + str(time_now))
    plt.show()
    plt.savefig(f'experiment_results/avg_reward_dqn_{time_now}.png')


