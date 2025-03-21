# Author: Changan Chen <changanvr@gmail.com>
# Modified by: Keyu Li <kyli@link.cuhk.edu.hk>

from __future__ import division
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super(ValueNetwork, self).__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)  

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL): 
    def __init__(self):
        super(SARL, self).__init__() 
        self.name = 'SARL'
        self.with_costmap = False
        self.gc = None
        self.gc_resolution = None
        self.gc_width = None
        self.gc_ox = None
        self.gc_oy = None
        self.max_value_list = list() ###########
        self.global_time = 0

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    # predict the cost that the robot hits the static obstacles in the global map
    def compute_cost(self, state):
        costs = []
        x = state.px
        y = state.py
        min_x = x - 0.1
        min_y = y - 0.1
        max_x = x + 0.1
        max_y = y + 0.1
        grid_min_x = int(round((min_x - self.gc_ox) / self.gc_resolution))
        grid_min_y = int(round((min_y - self.gc_oy) / self.gc_resolution))
        grid_max_x = int(round((max_x - self.gc_ox) / self.gc_resolution))
        grid_max_y = int(round((max_y - self.gc_oy) / self.gc_resolution))
        for i in range(grid_min_x, grid_max_x+1):
            for j in range(grid_min_y, grid_max_y + 1):
                index = i + self.gc_width * j
                costs.append(self.gc[index])
        max_cost = max(costs)
        return max_cost


    def predict(self, state):
        """
        Takes pairwise joint state as input to value network and output action.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length).
        If with_costmap is True, the dangerous actions predicted by the value network will be screened out to avoid static obstacles on the map.
        """
        #print(self.phase)
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
            #print("Action Space")
            #print(self.build_action_space(state.self_state.v_pref))

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
            #print("Max Action")
            #print(max_action)
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            '''#####################
            reward_dict = dict()
            value_dict = dict()
            r_list = []
            first_passed = False
            '''#####################
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)
                next_self_state_further = self.propagate_more(state.self_state, action)

                # abort actions which will probably cause collision with static obstacles in the costmap         
                if self.with_costmap is True:
                    cost = self.compute_cost(next_self_state_further)
                    if cost > 0:
                        print("********** Abort action:", action, " with cost:", cost, " that will hit the obstacles.")
                        continue
                

                if self.query_env:
                    next_human_states, reward, done, info, _, _,_,_ = self.env.onestep_lookahead(action)
                else:
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)
                    #reward = self.compute_reward_dg(state.self_state, next_self_state, next_human_states)
                #print(next_self_state)
                #next_self_state = torch.Tensor([next_self_state + (self.env.m_dist)])
                #print(next_self_state)
                #print(self.env.m_dist)
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                                   for next_human_state in next_human_states], dim=0)
                #print(batch_next_states)
                rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
                #print('INPUT RED NEURONAL')
                #print(rotated_batch_input)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps], dim=2)
                # VALUE UPDATE
                next_state_value = self.model(rotated_batch_input).data.item()
                #print('next state value = ', next_state_value)
                # Equation (5)
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                #print('Action-Value obatined: ')
                #print(value, action)
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
                    # print("********** choose action:", action)
                    # print("********** cost:", cost)
                '''#######################################
                if action.v != 0 or action.r != 0:
                    first_passed = True
                if not action.v in reward_dict:
                    reward_dict[action.v] = []
                    value_dict[action.v] = []
                if first_passed:
                    reward_dict[action.v].append(reward)
                    value_dict[action.v].append(value)
                    if not action.r in r_list:
                        r_list.append(action.r)

            if reward_dict and value_dict:
                #print('--------------------------reward_dict-----------------------------')
                #print(reward_dict)
                #print('--------------------------r_list-----------------------------')
                #print(r_list)
                reward_pd = pd.DataFrame(reward_dict, index=r_list)
                value_pd = pd.DataFrame(value_dict, index=r_list)

                print('------------------------------reward---------------------------------')
                print(reward_pd)
                print('----------------------------total value------------------------------')
                #print(value_pd)
                print("Chosen action:", max_action)
                print("Max value:", max_value)
                '''###########################################
            if max_action is None:
                # if the robot is trapped, choose the turning action to escape
                max_action = ActionRot(0, 0.78)
                #print("The robot is trapped. Rotate in place to escape......")
            
            #print('\nRobot exploits: max action:')
            #print(value, max_action)

        if self.phase == 'train':
            #print('if predict')
            self.last_state = self.transform(state)

        '''####################################
        self.max_value_list.append(max_value)
        print(self.max_value_list)
        '''####################################
        #print(max_action)
        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        if len(humans):
            for i, human in enumerate(humans):
                dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
                if dist < 0:
                    collision = True
                    break
                if dist < dmin:
                    dmin = dist

        # check if reaching the goal
        dg = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy))
        reaching_goal = dg < nav.radius

        if collision:
            reward = self.env.collision_penalty
        elif reaching_goal:
            reward = 1
        elif dmin < self.env.discomfort_dist:
            reward = (dmin - self.env.discomfort_dist) * self.env.discomfort_penalty_factor * self.env.time_step
        else:
            #reward = 0
            reward = 1 - 0.32*dg*dg # function 1
            # reward = 1 - 0.16*dg*dg # function 2
            # reward = 1 / (1 + np.exp(dg - 3)) #function 3
            #reward = 1 / (1 + np.exp(2.5*dg - 4)) #function 4
            # reward = 1 / (1 + np.exp(dg - 4)) #function 5

        return reward

    def compute_reward_dg(self, prev_nav, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        if len(humans):
            for i, human in enumerate(humans):
                dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
                if dist < 0:
                    collision = True
                    break
                if dist < dmin:
                    dmin = dist
        
        # compute dg at time t and t + 1
        start_dg = np.linalg.norm((prev_nav.px - prev_nav.gx, prev_nav.py - prev_nav.gy))
        end_dg = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy))

        # check if reaching the goal
        reaching_goal = end_dg < nav.radius

        # check if timeout
        Timeout = False
        self.global_time += self.time_step
        if self.global_time >= self.env.time_limit - 1 :
            Timeout = True

        if collision:
            reward = self.env.collision_penalty
        elif Timeout:
            reward = -10
        elif reaching_goal:
            reward = 10
        elif dmin < self.env.discomfort_dist:
            reward = (dmin - self.env.discomfort_dist) * self.env.discomfort_penalty_factor * self.env.time_step
            #reward = (dmin - self.env.discomfort_dist) * self.env.discomfort_penalty_factor
        else: 
            #reward = 0.01 * (start_dg - end_dg)
            reward = (1 - end_dg/start_dg) * self.env.time_step
            #reward =  - (end_dg*end_dg*end_dg*end_dg*end_dg)/(start_dg*start_dg*start_dg*start_dg*start_dg) * self.env.time_step

        return reward

    def get_attention_weights(self):
        return self.model.attention_weights
