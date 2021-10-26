'''
DQN Agent - Train an agent/DQN to learn a policy using value-based gradient.
Future: Try different methods (other value-based SGDs or policy-based)

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

Solution: Deep-Q Network
Train agent using an epsilon greedy method
-> Allow agent to act randomly for for n_timesteps (store history of state, action, reward)

pip install tensorflow==2.4.3
'''

import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


env = gym.make('LunarLander-v2')
env.seed(0)

class KerasDQN:

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = deque(maxlen=1000000)  # test changing this

        # agent parameters
        self.epsilon = 1
        self.epsilon_decay = 0.9 # how quickly to decay exploration vs. exploitation every episode
        self.epsilon_min = 0.01 # always explore at least 1% of the time
        self.gamma = 0.9 # how much you care about immediate vs. future rewards

        # nn parameters
        self.batch_size = 32
        self.learning_rate = 0.001 # TODO: Rename to alpha
        self.model = self.build_model()

    def build_model(self):

        model = keras.Sequential(
            [
                keras.Input(shape=(self.state_space)),
                layers.Dense(100, activation='relu', name='first_layer'), # input_size = state_size
                layers.Dense(50, activation='relu', name='second_layer'),
                layers.Dense(self.action_space, activation='linear', name='output_layer') # output_size = action_size
            ]
        )
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate)) # do we want to use ADAM?
        return model

    # store history
    def remember(self, state, action, reward, is_done, next_state):
        self.memory.append((state, action, reward, is_done, next_state))

    # choose an action using epsilon greedy method
    def act(self, state):
        if np.random.uniform(0,1) <= self.epsilon:
            action = random.randrange(self.action_space) # choose random action
        else:
            action = np.argmax(self.model.predict(state)) # choose greedy action
        return action

    # create batches from history to train
    def replay(self):

        if len(self.memory) < self.batch_size:
            return None # should I use break or continue or pass here

        minibatch = random.sample(self.memory, self.batch_size)
        state_batch = np.squeeze(np.array([i[0] for i in minibatch]))
        action_batch = np.array([i[1] for i in minibatch])
        reward_batch = np.array([i[2] for i in minibatch])
        is_done_batch = np.array([i[3] for i in minibatch])
        next_state_batch = np.squeeze(np.array([i[4] for i in minibatch]))

        # calculate a target we are trying to update our
        expected_return = reward_batch + \
                          self.gamma * \
                          np.amax(self.model.predict_on_batch(next_state_batch), axis=1) * \
                          (1 - is_done_batch) # what is this part doing?

        expected_return_full = self.model.predict_on_batch(state_batch)
        ind = np.array([i for i in range(self.batch_size)])
        expected_return_full[[ind], [action_batch]] = expected_return

        # fit model with minibatch
        self.model.fit(state_batch, expected_return, epochs=1, verbose=0)

        # decay the exploration until min
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def update(self, state, action, reward, next_state):
    #     # try to fit this update statement to replace the replay function above
    #
    #     q_target = (reward + self.gamma * np.min(self.model.predict(next_state)[0])) # why min here?
    #     q_values = self.model.predict(state)
    #     q_values[0][action] = q_target
    #     self.model.fit(state, q_values, verbose=0)


if __name__ == '__main__':

    print('------------------ Starting Lunar Lander Training ------------------ ')
    print('Number of actions: ', env.action_space.n)
    print('Observation space: ', env.observation_space.shape[0])
    print('-------------------------------------------------------------------- ')

    n_episodes = 2000
    max_frames_per_episode = 2000
    agent = KerasDQN(env.action_space.n, env.observation_space.shape[0]) #TODO: Add more parameters here -> learning rate ex

    experiment_info = 'Episodes= ' + str(n_episodes) +\
                      ' | max_frames= ' + str(max_frames_per_episode) +\
                      ' | learning_rate= ' + str(agent.learning_rate) +\
                      ' | batch_size= ' + str(agent.batch_size) +\
                      ' | gamma= ' + str(agent.gamma)

    print(f'\nTraining model over {n_episodes} episodes')

    all_episode_rewards = []

    for episode in range(n_episodes):

        #print('Running episode: ', episode)
        episode_reward = 0
        episode_reward_list = []
        num_frames = 1

        state = env.reset()
        state = state.reshape((1, -1))  # (1,8) -> is this right?

        for frame in range(max_frames_per_episode):

            action = agent.act(state) # should explore heavily in beginning of loops
            env.render()

            next_state, reward, is_done, _ = env.step(action)

            # store the information in the memory
            agent.remember(state, action, reward, is_done, next_state)  # is this going to be the right shape

            episode_reward += reward
            episode_reward_list.append(reward)
            num_frames += 1

            state = next_state
            state = state.reshape((1, -1))

            agent.replay()

            if is_done:
                print(f'Complete episode: {episode} | Total reward: {episode_reward} | Total frames: {num_frames}')
                all_episode_rewards.append(episode_reward_list)
                break

    # plot average and total rewards
    # tonight: change the rewards to be summed or averaged over the last 100 episodes
    # Run experiments change learning rate, epsilon decay etc
    # save the save_fig by experiment type (learning rate, etc - f string to store the information about the run)

    # TODO: future try to build on or change the reward function
    # maybe get more reward for staying upright?
    # maybe not penalize the main engine? Not sure why that is bad?

    # TODO: Maybe try to sample the memory of the best episodes to train on? (>top 30% of episodes)

    # TODO: Maybe try a policy algorithm 


    sum_reward_list = [np.sum(i) for i in all_episode_rewards]
    plt.plot([i + 1 for i in range(len(all_episode_rewards))], sum_reward_list)
    plt.title('Total rewards per episode \n' + str(experiment_info))
    plt.show()
    plt.savefig('total_reward_dqn_2.png') # TODO: make name dynamic

    avg_reward_list = [np.mean(i) for i in all_episode_rewards]
    plt.plot([i + 1 for i in range(len(all_episode_rewards))], avg_reward_list)
    plt.title('Average rewards per episode \n' + str(experiment_info))
    plt.show()
    plt.savefig('avg_reward_dqn_2.png') # TODO: make name dynamic




