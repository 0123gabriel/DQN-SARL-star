# DQN with SARL 

This project is based on [SARL](https://arxiv.org/pdf/1809.08835) (its implementation [here](https://github.com/vita-epfl/CrowdNav/tree/master)), [SARL*](https://ieeexplore.ieee.org/abstract/document/8961764) (its implementation [here](https://github.com/LeeKeyu/sarl_star))  and expands upon it by introducing new rewards, features, and environments to test the algorithm. The two new environments include both static and mobile agents, along with the robot.

The main goal of this project was to evaluate the algorithm in a race-like environment, such as F1TENTH. Some of the results are shown below:

https://github.com/user-attachments/assets/27886c58-4f9d-4438-ac30-7648813a59dd

https://github.com/user-attachments/assets/2717b5e9-24ee-4292-8d7e-055da641c2e8

## About the code

- ``` crowd_nav/ ``` – Contains all the files for training and testing the policy, along with configuration files for different environments.
- ``` crowd_sim/ ``` – Implements the environment dynamics and includes essential utility functions.
