# open_ai
 Range of RL projects using OpenAI gym package

### Autonomous Taxi 
Goal: Have a taxi agent learn how to navigate a grid world to pick up and drop off a passenger using tabular q-learning
* This notebook was created to teach coworkers about Q-learning

### Lunar Lander 
Goal: Teach a DQN agent to learn how to land a ship on a landing pad. 

#### <u><b>Experiments:</b></u>
* Memory size (memory): <i>experiment_results/avg_reward_dqn_11_23_2021_13_04.png</i>

<br><b>Observations:</b> Nearing 250 episodes, agents trained with both 100k and 1E6 experience replay buffer size observed better rolling average rewards. Nearly double 1E7 memory size. 

![alt text](https://github.com/megforr/open_ai/blob/main/lunar_lander/experiment_results/avg_reward_dqn_11_23_2021_13_04.png)

* Learning rate (alpha): <i>experiment_results/avg_reward_dqn_11_25_2021_01_53.png</i>

<br><b>Observations:</b> Agent learned in a more stable fashion at alpha=0.001. Maneuvers created by agent learning at 0.01 seemed much more risky (swinging wildly from side to side) opposed to the conservative upright landing style of the 0.001 agent. 

![alt text](https://github.com/megforr/open_ai/blob/main/lunar_lander/experiment_results/avg_reward_dqn_11_25_2021_01_53.png)

* Discount rate (gamma): <i>experiment_results/avg_reward_dqn_11_24_2021_20_27.png</i>

<br><b>Observations:</b>

![alt text](https://github.com/megforr/open_ai/blob/main/lunar_lander/experiment_results/avg_reward_dqn_11_24_2021_20_27.png)

Note: Best results would be observed if multiple experiments were performed and variance was calculated. 
