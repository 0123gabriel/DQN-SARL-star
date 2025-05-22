# Author: Changan Chen <changanvr@gmail.com>
# Modified by: Keyu Li <kyli@link.cuhk.edu.hk>

from __future__ import absolute_import
import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import torch
from matplotlib import patches
from numpy.linalg import norm

import sys
files_path = '/home/rise2/Gabriel/CrowdNav'
if files_path not in sys.path:
    sys.path.insert(0, files_path)
    
#print(sys.path)

from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import *
from crowd_sim.envs.utils.Astar import *
import random
import math
import sympy as sp

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.robot_path_length = 0  # @lky
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        self.heading_reward = None
        self.maintaining_reward = None
        self.standard_variation = None
        self.collision_wall_penalty = None
        self.R_safe = None
        self.R_min = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        # for block_area
        self.block_area1 = None
        self.block_area2 = None
        self.block_area3 = None
        # Shortest path
        self.short_path = None
        
        #block area for round road
        '''self.outer_circle = None
        self.lane_y_coord = None
        self.lane_x1_coord = None
        self.lane_x2_coord = None
        self.num_elements = None

        self.angles = None #np.linspace(np.arctan(lane_y_coord/lane_x1_coord), np.pi - np.arctan(lane_y_coord/lane_x1_coord), num_elements)
        self.outer_circle_radius = None #np.sqrt(np.power(lane_x1_coord, 2) + np.power(lane_y_coord, 2))
        self.inner_circle_radius = None
        self.circle_center = None'''

        #Block area por S path
        self.small_radius = None
        self.big_radius = None
        self.lower_center = None
        self.upper_center = None
        self.lower_circle_angles = None
        self.upper_circle_angles = None
        
        self.number_static_humans = None

        #Position where the robot starts and ends
        self.initial_robot_px = None
        self.initial_robot_py = None
        self.robot_gx = None
        self.robot_gy = None
        
        #Lists for the plot
        self.gif_w_points = None
        self.dist_orient_wp = None
        self.rob_wp_vectors = None
        self.xs = None
        self.ys = None
        self.curr_wp = None

    def set_human_num(self, human_num):
        self.human_num = human_num

    def set_humans(self, humans):
        self.humans = humans


    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.heading_reward = config.getfloat('reward', 'heading_reward')
        self.maintaining_reward = config.getfloat('reward', 'maintaining_reward')
        self.standard_variation = config.getfloat('reward', 'standard_variation')
        self.collision_wall_penalty = config.getfloat('reward', 'collision_wall_penalty')
        self.R_safe = config.getfloat('reward', 'R_safe')
        self.R_min = config.getfloat('reward', 'R_min')
        # Please input four points to make block area
        # Please input the center points, radius1, radius2
        self.block_area1 = [[-2,5],[-6,5],[-6,2],[-2,2]]
        self.block_area2 = [[6.5,-6],[2.5,-6],[2.5,-9],[6.5,-9]]
        self.block_area3 = [[8,-8],10,15]
        
        # Short path
        s_start = (10, -10)
        s_goal = (-11, 31)

        astar = AStar(s_start, s_goal, "euclidean")
        self.short_path, visited = astar.searching()
        #Block area for a round road
        '''self.outer_circle = []
        self.lane_y_coord = 3.5
        self.lane_x1_coord = 9.5
        self.lane_x2_coord = 22
        self.num_elements = 30

        self.angles = np.linspace(np.arctan(self.lane_y_coord/self.lane_x1_coord), np.pi - np.arctan(self.lane_y_coord/self.lane_x1_coord), self.num_elements)
        self.outer_circle_radius = np.sqrt(np.power(self.lane_x1_coord, 2) + np.power(self.lane_y_coord, 2))
        self.inner_circle_radius = 5
        self.circle_center = (0,0)'''
        
        #Block area for S path
        self.small_radius = 7.0
        self.big_radius = 13.6
        self.lower_center = (0,0)
        self.upper_center = (0, self.big_radius + self.small_radius)
        self.lower_circle_angles = ((1.0/2)*np.pi, (3.0/2)*np.pi)
        #self.angles_lower_circle = np.linspace((1.0/2)*np.pi, (3.0/2)*np.pi)

        #self.lower_small_circle =[]
        #self.lower_big_circle = []
        #self.upper_small_circle = []
        #self.upper_big_circle = []

        #self.min_x = 0.0
        #temp = 0.0

        #for i in range(len(self.angles_lower_circle) - 1):
        #    x1 = self.small_circle_radius*np.cos(self.angles_lower_circle[i])
        #    y1 = self.small_circle_radius*np.sin(self.angles_lower_circle[i])
        #    x2 = self.small_circle_radius*np.cos(self.angles_lower_circle[i+1])
        #    y2 = self.small_circle_radius*np.sin(self.angles_lower_circle[i+1])
        #    self.lower_small_circle.append([[x1,y1],[x2,y2]])

        #    x3 = self.big_circle_radius*np.cos(self.angles_lower_circle[i])
        #    y3 = self.big_circle_radius*np.sin(self.angles_lower_circle[i])
        #    x4 = self.big_circle_radius*np.cos(self.angles_lower_circle[i+1])
        #    y4 = self.big_circle_radius*np.sin(self.angles_lower_circle[i+1])
        #    self.lower_big_circle.append([[x3,y3],[x4,y4]])

        #    temp = min([x1, x2, x3, x4])
        #    self.min_x = min([temp, self.min_x])
            #print('self.min_x', self.min_x)

        #self.upper_circle_center = (0 + self.lower_big_circle[0][0][0], self.small_circle_radius + self.big_circle_radius)
        #self.angles_upper_circle = np.linspace(-(1.0/2)*np.pi, (1.0/2)*np.pi)
        self.upper_circle_angles = (-(1.0/2)*np.pi, (1.0/2)*np.pi)
        #self.max_x = 0.0
        #temp = 0.0

        #for i in range(len(self.angles_upper_circle) - 1):
        #    x5 = self.small_circle_radius*np.cos(self.angles_upper_circle[i]) + self.upper_circle_center[0]
        #    y5 = self.small_circle_radius*np.sin(self.angles_upper_circle[i]) + self.upper_circle_center[1]
        #    x6 = self.small_circle_radius*np.cos(self.angles_upper_circle[i+1]) + self.upper_circle_center[0]
        #    y6 = self.small_circle_radius*np.sin(self.angles_upper_circle[i+1]) + self.upper_circle_center[1]
        #    self.upper_small_circle.append([[x5,y5],[x6,y6]])

        #    x7 = self.big_circle_radius*np.cos(self.angles_upper_circle[i]) + self.upper_circle_center[0]
        #    y7 = self.big_circle_radius*np.sin(self.angles_upper_circle[i]) + self.upper_circle_center[1]
        #    x8 = self.big_circle_radius*np.cos(self.angles_upper_circle[i+1]) + self.upper_circle_center[0]
        #    y8 = self.big_circle_radius*np.sin(self.angles_upper_circle[i+1]) + self.upper_circle_center[1]
        #    self.upper_big_circle.append([[x7,y7],[x8,y8]])

        #    temp = max([x5, x6, x7, x8])
            #print('self.max_x', self.max_x)
        #    self.max_x = max(self.max_x, temp)
            #print('self.max_x', self.max_x)

        #print('max_x: ', self.max_x)

        self.number_static_humans = 2

        #Robot goal and initial position
        #self.initial_robot_px = random.uniform(self.lower_center[0] + self.robot.radius, self.big_radius - robot.radius) 
        #self.initial_robot_py = random.uniform(-self.big_radius, -self.small_radius)#self.lower_small_circle[-1][1][1] - self.small_circle_radius/2.0
        #self.final_robot_gx = -self.big_radius + 1
        #self.final_robot_gy = random.uniform(-self.big_radius, 2*self.big_radius - self.small_radius)#self.upper_small_circle[-1][1][1] + self.small_circle_radius/2.0
        
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    '''def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'static':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_static_human())
        elif rule == 'no':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_fake_human())
        elif rule == 'circle_static':
            self.humans = []
            for i in range(human_num):
                if i < 2:
                    self.humans.append(self.generate_static_human())
                else:
                    self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'square_static':
            self.humans = []
            for i in range(human_num):
                if i < 2:
                    self.humans.append(self.generate_static_human())
                else:
                    self.humans.append(self.generate_square_crossing_human())
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.humans = []
            if static:
                print("mode: static")
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    print("human num: 0, set fake human:(0, -10, 0, -10, 0, 0, 0)")
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                print("mode: dynamic")
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
            self.human_num = len(self.humans)
            self.human_times = [0] * self.human_num
            print("human number:", self.human_num)
        else:
            raise ValueError("Rule doesn't exist")'''
    
    def generate_random_human_position(self, human_num, rule):
        logging.info("human number: " + str(human_num))
        #logging.info(human_num)
        #print(human_num)
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'no':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_fake_human())
        #elif rule == 'static':
        #    self.humans = []
        #    for i in range(human_num):
        #        self.humans.append(self.generate_static_human())
        
        elif rule == 'circle_crossing':# before it was elif
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'circle_static':
            self.humans = []
            for i in range(human_num):
                if i < self.number_static_humans:
                    self.humans.append(self.generate_static_human())
                else:
                    self.humans.append(self.generate_circle_crossing_human())
        else:
            raise ValueError("Rule doesn't exist")

    def generate_fake_human(self):
        human = Human(self.config, 'humans')
        human.v_pref = 0
        while True:
            # add some noise to simulate all the possible cases robot could meet with human
            px = random.randint(40, 80)
            py = random.randint(60, 100)
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    print(collide)
                    break    # jump out of 'for' loop
            if not collide:
                break        # jump out of 'while' loop
        human.set(px, py, px, py, 0, 0, 0)
        return human


    def generate_static_human(self):
        print('Generating static humans')
        human = Human(self.config, 'humans')
        """if self.randomize_attributes:
            human.v_pref = human0.v_pref
            human.radius = human0.radius"""
        if self.randomize_attributes:
            human.random_radius()
        #print(human.radius)
        human.v_pref = 0
        while True:
            px = random.uniform(-self.big_radius, self.big_radius)
            py = random.uniform(-self.big_radius, 2*self.big_radius + self.small_radius)
            collide = False
            for agent in [self.robot] + self.humans:
                #the first 3 values let the robot not be outside the walls, and the last value leave enough space between 2 robots to let another one pass through
                min_dist = human.radius + agent.radius + self.discomfort_dist + 2*human.radius 
                if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break    # jump out of 'for' loop
            #dist = norm((px - , py - self.block_area3[0][1]))
            wall_dist = human.radius

            # Distance to the straight road at the bottom
            if py <= 0 and px >= 0:
                if point_to_segment_dist(self.lower_center[0], -self.small_radius, self.big_radius, -self.small_radius, px, py) < wall_dist or \
                        point_to_segment_dist(self.lower_center[0], -self.big_radius, self.big_radius, -self.big_radius, px, py) < wall_dist or py > -self.small_radius:
                            collide = True

            # Distance to the upper arc
            if py > 0 and px >= 0:
                if point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.small_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], px, py) < wall_dist or \
                        math.sqrt((self.upper_center[0] - px)**2 + (self.upper_center[1] - py)**2) < self.small_radius:
                            collide = True
                
                if point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.big_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], px, py) < wall_dist or \
                        math.sqrt((self.upper_center[0] - px)**2 + (self.upper_center[1] - py)**2) > self.big_radius:
                            collide = True

            # Distance to the lower arc
            if py < 21 and px < 0:
                if point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.small_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], px, py) < wall_dist or \
                        math.sqrt((self.lower_center[0] - px)**2 + (self.lower_center[1] - py)**2) < self.small_radius:
                            collide = True
                
                if point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.big_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], px, py) < wall_dist or \
                        math.sqrt((self.lower_center[0] - px)**2 + (self.lower_center[1] - py)**2) > self.big_radius:
                            collide = True

            # Distance to the straight road at the top
            if py >= 21 and px < 0:
                if point_to_segment_dist(self.upper_center[0], self.big_radius + 2*self.small_radius, -self.big_radius, self.big_radius + 2*self.small_radius, px, py) < wall_dist or \
                        point_to_segment_dist(self.upper_center[0], 2*self.big_radius + self.small_radius, -self.big_radius, 2*self.big_radius + self.small_radius, px, py) < wall_dist or \
                        py < self.big_radius + 2*self.small_radius or px < -self.big_radius + 5*human.radius + 1.5:
                            collide = True
            if not collide:
                break 
        human.set(px, py, px, py, 0, 0, 0)
        return human

    def generate_circle_crossing_human(self):
        print('Generating circle crossing humans')
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        #print(human.radius)
        #i = 0
        #np.random.seed(0)
        while True:
            #angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            '''px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = (self.circle_radius) * np.cos(angle) + px_noise
            py = (self.circle_radius) * np.sin(angle) + py_noise'''
            #px = (self.circle_radius) * np.cos(angle) + px_noise
            #py = (self.circle_radius) * np.sin(angle) + py_noise
            #px = self.max_x - 1 # 21 #random.uniform(-6, 6)
            #py = random.uniform(-self.big_circle_radius, -self.small_circle_radius) #random.uniform(-3.5, 3.5)
            px = random.uniform(self.lower_center[0] + human.radius, self.big_radius - human.radius)
            py = random.uniform(-self.small_radius - human.radius, -self.big_radius + human.radius)
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5

                if norm((px - agent.px, py - agent.py)) < min_dist or  norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
                    # jump out of 'for' loop

            wall_dist = human.radius + self.discomfort_dist + 0.15

            if point_to_segment_dist(self.lower_center[0], -self.small_radius, self.big_radius, -self.small_radius, px, py) < wall_dist or \
                    point_to_segment_dist(self.lower_center[0], -self.big_radius, self.big_radius, -self.big_radius, px, py) < -wall_dist:
                collide = True

            if not collide:
                break 

        # 5 is the first y-coordinate goal of the robots for this S-path scenario only
        cent_dist = 3
        
        human_gy = -cent_dist
        
        x = sp.Symbol('x')
        eq = (((self.small_radius**2)*x**2/(x**2+(human_gy**2)))**(1.0/2) - x)**2 + ((human_gy/x)*((self.small_radius**2)*x**2/(x**2+(human_gy**2)))**(1.0/2) - human_gy)**2 - (self.robot.radius)**2
        human_gx_min = max(sp.solve(eq, x))
        
        eq = (((self.big_radius**2)*x**2/(x**2+(human_gy**2)))**(1.0/2) - x)**2 + ((human_gy/x)*((self.big_radius**2)*x**2/(x**2+(human_gy**2)))**(1.0/2) - human_gy)**2 - (self.robot.radius)**2
        human_gx_max = min(sp.solve(eq, x))
        
        human.set(px, py, -float(random.uniform(human_gx_min, human_gx_max)), human_gy, 0, 0, 0)
        #print(human.get_goal_position())
        return human        # jump out of 'while' loop
        

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        sign = 0
        while True:
            #px = float(random.sample((-self.circle_radius-3, self.circle_radius+3),1)[0])
            #py = random.uniform(-self.circle_radius-3 * 1.5 / 2.0, self.circle_radius+3 * 1.5 / 2.0)
            px = float(random.sample((-self.circle_radius, self.circle_radius),1)[0])
            py = random.uniform(-self.circle_radius * 1.5 / 2.0, self.circle_radius * 1.5 / 2.0)
            collide = False
            for agent in [self.robot] + self.humans:
                sign += 1
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break    # jump out of 'for' loop
            if px == self.circle_radius and (py<=self.block_area1[0][1]+human.radius and py>=self.block_area1[2][1]-human.radius):
                collide = True
            elif px == -self.circle_radius and (py<=self.block_area2[0][1]+human.radius and py>=self.block_area2[2][1]-human.radius):
                collide = True
            if px == self.circle_radius:
                gx = self.block_area2[0][0] + human.radius + 1
            else:
                gx = self.block_area1[2][0] - human.radius - 1
            gy = py
            if sign % 2 == 0:
                temp1 = px
                px = py
                py = temp1
                temp2 = gx
                gx = gy
                gy = temp2
            if not collide:
                break        # jump out of 'while' loop
        '''while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break'''
        human.set(px, py, gx, gy, 0, 0, 0)
        return human
    
    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, params[0],params[1],params[2],params[3], 0.3, 1)
        sim.addAgent(self.robot.get_position(), params[0],params[1],params[2],params[3], self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), params[0],params[1],params[2],params[3], human.radius, human.v_pref, human.get_velocity())


        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if human.reached_destination():
                    self.human_times[i] = self.global_time
                    """if self.randomize_attributes:
                        goal = human.get_goal_position() # makes humans move continuously
                        new_goal = [-goal[0],-goal[1]]
                        human.set_goal_position(new_goal)
                        self.human_times[i] = 0"""
                    if self.train_val_sim == 'square_crossing' or self.test_sim == 'square_crossing':
                        goal = human.get_goal_position()
                        if goal[0] == self.block_area1[2][0] - human.radius - 1:
                            new_goal = [-self.circle_radius,goal[1]]
                        elif goal[0] == self.block_area2[0][0] + human.radius + 1:
                            new_goal = [self.circle_radius, goal[1]]
                        elif goal[1] == self.block_area1[2][1] - human.radius - 1:
                            new_goal = [goal[0], -self.circle_radius]
                        else:
                            new_goal = [goal[0], self.circle_radius]
                        human.set_goal_position(new_goal)
                    if self.train_val_sim == 'circle_crossing' or self.test_sim == 'circle_crossing':
                        goal = human.get_goal_position()
                        new_goal = [goal[1],goal[0]]
                        human.set_goal_position(new_goal)
                    elif self.train_val_sim == 'square_static' or self.test_sim == 'square_static':
                        if i >= 2:
                            goal = human.get_goal_position()
                            if goal[0] == self.circle_radius:
                                new_goal = [-goal[0],goal[1]]
                            else:
                                new_goal = [goal[0],-goal[1]]
                            human.set_goal_position(new_goal)
                    elif self.train_val_sim == 'circle_static' or self.test_sim == 'circle_static':
                        if i >= 2:
                            goal = human.get_goal_position()
                            if goal[0] == self.block_area1[2][0] - human.radius - 1:
                                new_goal = [-self.circle_radius,goal[1]]
                            elif goal[0] == self.block_area2[0][0] + human.radius + 1:
                                new_goal = [self.circle_radius, goal[1]]
                            elif goal[1] == self.block_area1[2][1] - human.radius - 1:
                                new_goal = [goal[0], -self.circle_radius]
                            else:
                                new_goal = [goal[0], self.circle_radius]
                            human.set_goal_position(new_goal)

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, number, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        
        self.gif_w_points = []
        self.dist_orient_wp = []
        self.rob_wp_vectors = []
        self.xs = []
        self.ys = []
        
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'
        if number %3 == 0:
            self.train_val_sim = 'no'
            self.test_sim = 'no'
        #elif number %4 == 1:
        #    self.train_val_sim = 'static'
        #    self.test_sim = 'static'
        elif number %3 == 1:#before it was elif
            self.train_val_sim = 'circle_crossing'
            self.test_sim = 'circle_crossing'
        elif number %3 == 2:
            self.train_val_sim = 'circle_static'
            self.test_sim = 'circle_static'
        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.initial_robot_px = 9.0 #round(random.uniform(self.lower_center[0] + self.robot.radius + 0.5, self.big_radius - self.robot.radius - 0.5), 2) #10
            self.initial_robot_py = -11.0 #round(random.uniform(-self.small_radius - 2*self.robot.radius, -self.big_radius + 2*self.robot.radius), 2) #-10
            self.robot_gy = 31.0 #round(random.uniform(self.big_radius + 2*self.small_radius + self.robot.radius + 0.5, 2*self.big_radius + self.small_radius - self.robot.radius - 0.5)) #-8 #random.uniform(self.big_radius + 2*self.small_radius + self.robot.radius, 2*self.big_radius + self.small_radius - self.robot.radius)
            self.robot_gx = -self.big_radius + 3.6 #11 #random.uniform(-np.sqrt((self.small_radius + self.robot.radius + self.discomfort_dist)**2 - (robot_gy - self.lower_center[1])**2) + self.lower_center[0], -np.sqrt((self.big_radius - self.robot.radius - self.discomfort_dist)**2 - (robot_gy - self.lower_center[1])**2) + self.lower_center[0])#-self.big_radius + 3.6
            
            md1 = point_to_segment_dist(self.lower_center[0], -self.small_radius, self.big_radius, -self.small_radius, self.initial_robot_px, self.initial_robot_py) 
            md2 = point_to_segment_dist(self.lower_center[0], -self.big_radius, self.big_radius, -self.big_radius, self.initial_robot_px, self.initial_robot_py)
            min_distance = min(md1, md2)
            #print('Min distance: ', min_distance)
            # Short path
            s_start = (self.initial_robot_px, self.initial_robot_py)
            ##print('Robot start position: ', s_start)
            s_goal = (self.robot_gx, self.robot_gy)

            astar = AStar(s_start, s_goal, "euclidean")
            self.short_path, visited = astar.searching()
            self.number_of_steps = 40 # 1 Means that the only existing way point is the goal
            step_len = len(self.short_path)//self.number_of_steps
            self.w_points = []
            
            for i in range(1, self.number_of_steps):
                self.w_points.append(self.short_path[i*step_len])

            self.w_points = self.w_points[::-1]
            self.w_points.append(s_goal)
            self.curr_wp = len(self.w_points)
            print('Length of w_points:', self.curr_wp)#
            self.all_w_points = self.w_points[:]
            
            if len(self.w_points) >= 1:
                self.robot_direction = get_agent_direction(np.array(self.w_points[0]), np.array([self.initial_robot_px, self.initial_robot_py]))
            else:
                self.robot_direction = get_agent_direction(np.array([self.robot_gx, self.robot_gy]), np.array([self.initial_robot_px, self.initial_robot_py]))
            ##print('Robot Direction {}'.format(self.robot_direction))
            ##print('Short Path Lenght: ', len(self.short_path))
            ##print('Way points selected: ', [[round(point[0], 2), round(point[1], 2)] for point in self.w_points])
            ##print('Robot Goal Position: ', robot_gx, robot_gy)
            #self.robot.set(self.initial_robot_px, self.initial_robot_py, robot_gx, robot_gy, 0, 0, np.pi / 2)
            self.robot.set(self.initial_robot_px, self.initial_robot_py, self.w_points[0][0], self.w_points[0][1], 0, 0, np.pi / 2, min_distance)
            #self.gif_w_points.append(self.w_points)
            #self.dist_orient_wp.append((self.robot_direction, norm(self.w_points[0][0] - self.initial_robot_px) if self.robot_direction == 'h' else norm(self.w_points[0][1] - self.initial_robot_py)))
            #print('Min distance saved', self.robot.min_dist)
            #print('Robot: ', self.robot)
            
            if self.case_counter[phase] >= 0:
                #np.random.seed(counter_offset[phase] + self.case_counter[phase]) #training
                np.random.seed(counter_offset[phase] + self.case_counter[phase]) #testing in the same scenarios 
                random.seed(counter_offset[phase] + self.case_counter[phase]) #testing in the same scenarios
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)

                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        #print('One step look ahead called')
        if len(self.w_points) != self.curr_wp:
            print('Length of w_points:', len(self.w_points))#
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))
    
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # Collision with walls
        collision_wall = False
        if self.robot.kinematics == 'holonomic':
            next_px = self.robot.px + action.vx * self.time_step
            next_py = self.robot.py + action.vy * self.time_step
        else:
            next_px = self.robot.px + action.v * np.cos(action.r + self.robot.theta) * self.time_step
            next_py = self.robot.py + action.v * np.sin(action.r + self.robot.theta) * self.time_step

        min_dist = self.robot.radius
        
        # Distance to the straight road at the bottom
        if next_py <= 0 and next_px >= 0:
            if point_to_segment_dist(self.lower_center[0], -self.small_radius, self.big_radius, -self.small_radius, next_px, next_py) < min_dist or \
                    point_to_segment_dist(self.lower_center[0], -self.big_radius, self.big_radius, -self.big_radius, next_px, next_py) < min_dist or \
                        next_py > -self.small_radius or next_py < -self.big_radius or next_px > self.big_radius:
                        collision_wall = True
                        #print("botoom")
                        #print(next_px, next_py)

        # Distance to the upper arc
        if next_py > 0 and next_px >= 0:
            if point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.small_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], next_px, next_py) < min_dist or \
                    math.sqrt((self.upper_center[0] - next_px)**2 + (self.upper_center[1] - next_py)**2) < self.small_radius:
                        collision_wall = True
                        #print("small upper arc")
                        #print(next_px, next_py)

            if point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.big_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], next_px, next_py) < min_dist or \
                    math.sqrt((self.upper_center[0] - next_px)**2 + (self.upper_center[1] - next_py)**2) > self.big_radius:
                        collision_wall = True
                        #print("big upper arc")
                        #print(point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.big_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], next_px, next_py) < min_dist)
                        #print(math.sqrt((self.upper_center[0] - px)**2 + (self.upper_center[1] - py)**2))
                        #print(self.big_radius)
                        #print(math.sqrt((self.upper_center[0] - px)**2 + (self.upper_center[1] - py)**2) > self.big_radius)
                        #print(next_px, next_py)

        # Distance to the lower arc
        if next_py < 21 and next_px < 0:
            if point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.small_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], next_px, next_py) < min_dist or \
                    math.sqrt((self.lower_center[0] - next_px)**2 + (self.lower_center[1] - next_py)**2) < self.small_radius:
                        collision_wall = True
                        #print("small lower arc")
                        #print(next_px, next_py)

            if point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.big_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], next_px, next_py) < min_dist or \
                    math.sqrt((self.lower_center[0] - next_px)**2 + (self.lower_center[1] - next_py)**2) > self.big_radius:
                        collision_wall = True
                        #print("big lower arc")
                        #print(next_px, next_py)

        # Distance to the straight road at the top
        if next_py >= 21 and next_px < 0:
            if point_to_segment_dist(self.upper_center[0], self.big_radius + 2*self.small_radius, -self.big_radius, self.big_radius + 2*self.small_radius, next_px, next_py) < min_dist or \
                    point_to_segment_dist(self.upper_center[0], 2*self.big_radius + self.small_radius, -self.big_radius, 2*self.big_radius + self.small_radius, next_px, next_py) < min_dist:# or \
                    #next_py < self.big_radius + 2*self.small_radius or next_px < -self.big_radius + 5*self.humans[0].radius + 1.5:
                        collision_wall = True
                        #print("top")
                        #print(next_px, next_py)

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if reaching the goal
        #end_position = np.array(self.robot.compute_position(action, self.time_step))
        ##start_dg = norm(self.robot.get_position() - np.array(self.robot.get_goal_position()))
        #end_dg = norm(end_position - np.array(self.robot.get_goal_position()))
        #reaching_goal = end_dg < self.robot.radius
        #position_variation = norm(end_position - self.robot.get_position())
        #reward = 0
	
        # check if another another robot arrive the goal
        #another_won = False
        #for i, human in enumerate(self.humans):
        #    print(self.static_humans)
        #    if i >= self.static_humans:
        #        another_won = norm(np.array(human.get_position()) - np.array(human.get_goal_position())) < human.radius
                #end_position = np.array(human.compute_position(action, self.time_step))
                #start_dg = norm(human.get_position() - np.array(human.get_goal_position())) < human.radius
                #end_dg = norm(end_position - np

	    # calculate the closest distance between robot and humans
        static_dmin = float('inf')
        dynamic_dmin = float('inf')
        R_danger = 0
        R_goal = 0
        R_collision = 0
        R_stop = 0
        dot_prod = 0
        R_col_wall = 0
        timeout = 0
        for i, human in enumerate(self.humans):
            if (human.vx == 0 and human.vy == 0):
                s_px = human.px - self.robot.px
                s_py = human.py - self.robot.py
                if self.robot.kinematics == 'holonomic':
                    s_vx = human.vx - action.vx
                    s_vy = human.vy - action.vy
                else:
                    s_vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                    s_vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
                s_ex = s_px + s_vx * self.time_step
                s_ey = s_py + s_vy * self.time_step
                # closest distance between boundaries of two agents
                static_closest = point_to_segment_dist(s_px, s_py, s_ex, s_ey, 0, 0) - human.radius - self.robot.radius
                if static_closest < static_dmin:
                    static_dmin = static_closest
            else:
                d_px = human.px - self.robot.px
                d_py = human.py - self.robot.py
                if self.robot.kinematics == 'holonomic':
                    d_vx = human.vx - action.vx
                    d_vy = human.vy - action.vy
                else:
                    d_vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                    d_vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
                d_ex = d_px + d_vx * self.time_step
                d_ey = d_py + d_vy * self.time_step
                # closest distance between boundaries of two agents
                dynamic_closest = point_to_segment_dist(d_px, d_py, d_ex, d_ey, 0, 0) - human.radius - self.robot.radius
                if dynamic_closest < dynamic_dmin:
                    closest_index = i
                    dynamic_dmin = dynamic_closest
                clo_rel_po = [self.robot.px - self.humans[closest_index].px, self.robot.py - self.humans[closest_index].py]
                clo_rel_ve = [self.robot.vx - self.humans[closest_index].vx, self.robot.vy - self.humans[closest_index].vy]
                dot_prod = np.dot(clo_rel_po, clo_rel_ve)
        
        #Update the way points in self.w_points and save its value before the elimination for R_wp calculation
        self.curr_wp = len(self.w_points)
        ##print('Robot Direction {}'.format(self.robot_direction))
        #self.gif_w_points.append(self.w_points)
        if len(self.w_points) >= 1 and update:
            self.dist_orient_wp.append((self.robot_direction, abs(self.robot.get_position()[0] - self.w_points[0][0]) if self.robot_direction == 'h' else abs(self.robot.get_position()[1] - self.w_points[0][1])))
            
            if self.robot_direction == 'h':
                if abs(self.robot.get_position()[0] - self.w_points[0][0]) < self.robot.v_pref*self.time_step + self.robot.radius + 1.2: 
                    # v_pref*time_step is the max distance a robot can reach a sub goal (way point) in one step, + robot.radius takes into account the farest point of the robot and +1.5 is a clearance 
                    ##print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
                    self.w_points = self.w_points[1:] if len(self.w_points) >= 2 else self.w_points[:]
                    self.robot_direction = get_agent_direction(np.array(self.w_points[0]), np.array(self.robot.get_position())) 
            
            if self.robot_direction == 'v':
                if abs(self.robot.get_position()[1] - self.w_points[0][1]) < self.robot.v_pref*self.time_step + self.robot.radius + 1.2:
                    ##print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
                    self.w_points = self.w_points[1:] if len(self.w_points) >= 2 else self.w_points[:]
                    self.robot_direction = get_agent_direction(np.array(self.w_points[0]), np.array(self.robot.get_position()))
            
            self.gif_w_points.append(self.w_points)
            self.rob_wp_vectors.append((self.robot.get_position(), self.w_points[0]))
            
            rwp_vec = np.array(self.w_points[0]) - np.array(self.robot.get_position())
            self.xs.append(abs(rwp_vec[0]))
            self.ys.append(abs(rwp_vec[1]))
        
        if len(self.w_points) < 1 and update:
            self.dist_orient_wp.append(('g', abs(self.robot.get_position()[0] - self.robot.get_position_goal()[0][0]) if self.robot_direction == 'h' else abs(self.robot.get_position()[1] - self.robot.get_position_goal()[0][1])))
            self.gif_w_points.append(self.w_points)
            self.rob_wp_vectors.append((self.robot.get_position(), self.robot.get_goal_position()))
            
            rwp_vec = np.array(self.robot.get_goal_position()) - np.array(self.robot.get_position())
            self.xs.append(abs(rwp_vec[0]))
            self.ys.append(abs(rwp_vec[1]))

        # check if reaching the goal
        end_position = np.array(self.robot.compute_position(action, self.time_step))
        start_dg = norm(self.robot.get_position() - np.array(self.robot.get_goal_position()))
        end_dg = norm(end_position - np.array((self.robot_gx, self.robot_gy))) # self.robot.get_goal_position()
        reaching_goal = end_dg < self.robot.radius
        position_variation = norm(end_position - self.robot.get_position())
        reward = 0
        
        #self.gif_w_points.append(self.w_points)
        #self.rob_wp_vectors.append((self.robot.get_position(), self.w_points[0]))

        # Check if the robot follows the shortest path while getting closer to every way point
        ##print('Way points len: ', len(self.w_points))
        ##print('Way points: ', [[round(point[0], 2), round(point[1], 2)] for point in self.w_points])
        R_way_point_dist = 0
        
        next_pos = np.array(self.robot.compute_position(action, self.time_step))
        curr_pos = np.array(self.robot.get_position())
        
        if len(self.w_points) >= 1:
            next_wp = np.array(self.w_points[0])
        else:
            next_wp = np.array((self.robot_gx, self.robot_gy))

        dist_next_p = norm(next_pos - next_wp)
        dist_curr_p = norm(curr_pos - next_wp)

        if dist_next_p < dist_curr_p:
            R_way_point_dist = 0.03
        else:
            R_way_point_dist = -0.04

        # Check if the robot follows the direction to the closest way point
        R_way_point_dir = 0

        # Unit vector from the robot position to the next way point 
        rob_wp_uni_vector = get_unit_vector(np.array(self.w_points[0]), np.array(self.robot.get_position()))
        rob_dir_uni_vector = get_agent_unit_vector(self.robot)
        R_way_point_dir = get_dir_wp_reward(rob_wp_uni_vector, rob_dir_uni_vector)*0.1
        ##print('Unit Vector to closest way point: ', rob_wp_uni_vector)
        ##print('Unit Vector in the rebots direction: ', rob_dir_uni_vector)
        ##print('Reward to follow the way point: ', R_way_point_dir)

        # check whether robot is stopped
        is_stopped = False
        R_stop_t = 0
        if self.robot.kinematics == 'holonomic':
            if norm([action.vx, action.vy]) < 0.01:
                is_stopped = True
                R_stop_t = -0.01
        else:
            if action.v < 0.01:
                is_stopped = True
                R_stop_t = -0.01


        r_danger_r = False
        if static_dmin <= self.R_safe:
            R_danger = -0.03
            r_danger_r = True

        if dynamic_dmin < self.R_min:
            R_danger = -0.03
            r_danger_r = True

        if not r_danger_r:
            R_danger = 0.05

        left_path = 0
        if len(self.w_points) > 1:
            left_path = norm(np.array(self.w_points[0]) - np.array(self.robot.get_position())) - self.robot.radius #end_dg - self.robot.radius
            for i in range(len(self.w_points) - 1):
                left_path = left_path + norm(np.array(self.w_points[i]) - np.array(self.w_points[i+1]))

        else:
            left_path = norm(np.array(self.robot.get_position()) - np.array(self.robot.get_goal_position())) - self.robot.radius
        position = self.robot.get_position()
        R_wall_min_dist = 0
        min_wall_bool = False

        # Distance to the walls of the bottom straight road
        if position[1] <= 0 and position[0] >= 0:
            ##print("Dist to the bottom")
            d1 = point_to_segment_dist(self.lower_center[0], -self.small_radius, self.big_radius, -self.small_radius, position[0], position[1])
            d2 = point_to_segment_dist(self.lower_center[0], -self.big_radius, self.big_radius, -self.big_radius, position[0], position[1])

        # Distance to the walls of the upper arc
        if position[1] > 0 and position[0] >= 0:
            d1 = point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.small_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], position[0], position[1])
            d2 = point_to_arc_dist(self.upper_center[0], self.upper_center[1], self.big_radius, self.upper_circle_angles[0], self.upper_circle_angles[1], position[0], position[1])
        
        # Distance to the walls of the lower arc
        if position[1] < 21 and position[0] < 0:
            ##print("Dist to the lower arcs")
            d1 = point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.small_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], position[0], position[1])
            d2 = point_to_arc_dist(self.lower_center[0], self.lower_center[1], self.big_radius, self.lower_circle_angles[0], self.lower_circle_angles[1], position[0], position[1])

        # Distance to the straight road at the top
        if position[1] >= 21 and position[0] < 0:
            d1 = point_to_segment_dist(self.upper_center[0], self.big_radius + 2*self.small_radius, -self.big_radius, self.big_radius + 2*self.small_radius, position[0], position[1])
            d2 = point_to_segment_dist(self.upper_center[0], 2*self.big_radius + self.small_radius, -self.big_radius, 2*self.big_radius + self.small_radius, position[0], position[1]) 


        #d1 = math.sqrt((self.block_area3[0][0] - position[0])**2 + (self.block_area3[0][1] - position[1])**2) - self.block_area3[1]
        #d2 = self.block_area3[2] - math.sqrt((self.block_area3[0][0] - position[0])**2 + (self.block_area3[0][1] - position[1])**2)
        #print('Robot Position: ', self.robot.get_position())
        #print('Robot Goal: ', self.robot.get_goal_position())
        ##print("Distancia 1: ", d1)
        ##print("Distancia 2: ", d2)
        min_distance = min(d1, d2)
        min_distance -= self.robot.radius
        m_dist = min_distance
        r_wall_dist_reward = lambda x : np.exp(x+1.8) - 26.8
        if min_distance < 1.5:
            R_wall_min_dist = r_wall_dist_reward(min_distance)/100.0
            min_wall_bool = True
        else:
            R_wall_min_dist = 0.0025
            
        #R_forward = 0.1/(1 + math.exp(len(self.w_points) - 5)) if len(self.w_points) < 10 else -0.1
        value_per_wp = 0.015
        wp_func = lambda x : -value_per_wp*(x - self.number_of_steps) # 2.25 is the maximum reward when ther is 1 way point left
        R_wp = wp_func(len(self.w_points)) if self.curr_wp != len(self.w_points) else 0
        #R_forward = ((1.0/3)*(math.exp(-(1.0/2)*len(self.w_points)) + 3.0) - 0.07)*0.05
	
        #if end_dg - start_dg < -0.01:
        #    R_forward = 0.01
        #else:
        #    R_forward = -0.01
	
	    #check if robot keeps moving
        new_position = np.array(self.robot.compute_position(action, self.time_step))
        position_variation = norm(new_position - self.robot.get_position())

        if position_variation > 0.2:
            R_km = 0.05
        else:
	        R_km = -0.5

        if reaching_goal:
            done = True
            info = ReachGoal()
            R_goal = self.success_reward
            left_path = 0
        elif collision_wall:
            done = True
            R_col_wall = self.collision_wall_penalty
            info = CollisionWall()
	        #R_for = 3*(1 / (1+np.exp(3*end_dg-5.5)))**0.1
        elif collision:
            done = True
            info = Collision()
            R_collision = self.collision_penalty
	        #R_for += 3*(1 / (1+np.exp(3*end_dg-5.5)))**0.1
            #elif another_won:
           #    done = True
            #    info = Loser()
        elif self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
            timeout = -25
	        #R_for += 3*(1 / (1+np.exp(3*end_dg-5.5)))**0.1
        else:
            done = False
            info = Nothing()
        
        reward = R_danger + R_goal + R_collision + R_km + R_col_wall + R_wall_min_dist + R_way_point_dist + R_wp + timeout
        reward_values = {"Total Reward": reward,"R_dan": R_danger, "R_goal": R_goal,"R_col": R_collision, "R_km": R_km, "R_col_wall": R_col_wall, \
                        "R_wall_min_dist": R_wall_min_dist, "R_way_point_dist": R_way_point_dist, \
                        "R_wp": R_wp, "timeout": timeout} 
        
        if update:
            # store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # update all agents
            # Commented zone is to calculate the shortest distance to the goal, which is a line
            #segment_x1 = -self.lane_x2_coord + 1 #-21
            #segment_x2 = -self.lane_x2_coord + 1 #-21
            #segment_y1 = self.lane_y_coord #3.5
            #segment_y2 = -self.lane_y_coord #3.5
            #posr_x, posr_y = self.robot.get_position()
            #goal_x, goal_y = nearest_point_on_segment(segment_x1, segment_y1, segment_x2, segment_y2, posr_x, posr_y)
            self.robot.step(action, m_dist)#, goal_x, goal_y)
            if self.robot.get_goal_position()[0] != self.w_points[0][0] or self.robot.get_goal_position()[1] != self.w_points[0][1] and self.w_points is not None:
                ##print('Cambio next way point')
                self.robot.set_goal_position((self.w_points[0][0], self.w_points[0][1]))
                ##print(self.robot.get_goal_position())
            if self.w_points is None:
                ##print('Cambio a meta final')
                self.robot.set_goal_position((-self.big_radius + 2.6, 29.0))
            # Change goal position of the robot
            #r_pos = self.robot.get_position()
            #if r_pos[0] < -4 and r_pos[0] > -5: #r_pos[1] > -9 and r_pos[1] < -8:
            #    robot_gy = 5
            #    self.robot.set_goal_position((random.uniform(np.sqrt((self.small_radius + self.robot.radius + self.discomfort_dist)**2 - (robot_gy - self.lower_center[1])**2) + self.lower_center[0], np.sqrt((self.big_radius - self.robot.radius - self.discomfort_dist)**2 - (robot_gy - self.lower_center[1])**2) + self.lower_center[0]), robot_gy))
            #    print(self.robot.get_goal_position())

            #if r_pos[1] > 2 and r_pos[1] < 4:
            #    robot_gy = 14
            #    self.robot.set_goal_position((random.uniform(np.sqrt((self.small_radius + self.robot.radius + self.discomfort_dist)**2 - (robot_gy - self.upper_center[1])**2) + self.upper_center[0], np.sqrt((self.big_radius - self.robot.radius - self.discomfort_dist)**2 - (robot_gy - self.upper_center[1])**2) + self.upper_center[0]), robot_gy))
            #    print(self.robot.get_goal_position())

            #if r_pos[1] > 11 and r_pos[1] < 13:
            #    robot_gy = 30
            #    self.robot.set_goal_position((random.uniform(np.sqrt((self.small_radius + self.robot.radius + self.discomfort_dist)**2 - (robot_gy - self.upper_center[1])**2) + self.upper_center[0], np.sqrt((self.big_radius - self.robot.radius - self.discomfort_dist)**2 - (robot_gy - self.upper_center[1])**2) + self.upper_center[0]), robot_gy))
            #    print(self.robot.get_goal_position())

            #if r_pos[1] > 27 and r_pos[1] < 29:
            #    self.robot.set_goal_position((-self.big_radius + 3.6, random.uniform(self.big_radius + 2*self.small_radius + self.robot.radius + self.discomfort_dist, 2*self.big_radius + self.small_radius - self.robot.radius - self.discomfort_dist)))
            #    print(self.robot.get_goal_position())

            #print(self.robot.get_goal_position())
            for i, human_action in enumerate(human_actions):
                #posh_x, posh_y = self.humans[i].get_position()
                #goalh_x, goalh_y = nearest_point_on_segment(segment_x1, segment_y1, segment_x2, segment_y2, posh_x, posh_y)
                self.humans[i].step(human_action)#, goalh_x, goalh_y)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                pos = human.get_position()
                if pos[1] > -5 and pos[1] < -3:
                    cent_dist = 3
                    human.set_goal_position((abs(human.get_goal_position()[0]), self.big_radius + self.small_radius + cent_dist))
                    #print(human.get_goal_position())

                if pos[1] > 18 and pos[1] < 20:
                    human.set_goal_position((-self.big_radius + 0.5, random.uniform(self.big_radius + 2*self.small_radius + human.radius, 2*self.big_radius + self.small_radius - human.radius)))
                #if i >= self.number_static_humans:
                    #print('human pos', human.get_position())
                #    print('human goal', human.get_goal_position())
                # Gabriel's Reminder: delete this commented section
                #if i >= self.static_humans:
                #    if human.get_position()[0] < self.min_x + 1:
                #        done = True
                #        info = Loser()
                if pos[0] < -12.4:
                    pos = human.get_position()
                    human.set_goal_position(pos)
                    goal = human.get_goal_position()
                    human.set_position(goal)
                # only record the first time the human reaches the goal
                if human.reached_destination():
                    self.human_times[i] = self.global_time
                    #done = True
                    #info = Loser()
                    """if self.randomize_attributes:
                        goal = human.get_goal_position() # makes humans move continuously
                        new_goal = [-goal[0],-goal[1]]
                        human.set_goal_position(new_goal)
                        self.human_times[i] = 0"""
                    if self.train_val_sim == 'square_crossing' or self.test_sim == 'square_crossing':
                        goal = human.get_goal_position()
                        if goal[0] == self.block_area1[2][0] - human.radius - 1:
                            new_goal = [-self.circle_radius,goal[1]]
                        elif goal[0] == self.block_area2[0][0] + human.radius + 1:
                            new_goal = [self.circle_radius, goal[1]]
                        elif goal[1] == self.block_area1[2][1] - human.radius - 1:
                            new_goal = [goal[0], -self.circle_radius]
                        else:
                            new_goal = [goal[0], self.circle_radius]
                        human.set_goal_position(new_goal)
                    if self.train_val_sim == 'circle_crossing' or self.test_sim == 'circle_crossing':
                        # Keep the mobile robots still
                        goal = human.get_goal_position()
                        new_goal = [goal[0], goal[1]]
                        human.set_position(new_goal)
                    elif self.train_val_sim == 'square_static' or self.test_sim == 'square_static':
                        if i >= 2:
                            goal = human.get_goal_position()
                            if goal[0] == self.circle_radius:
                                new_goal = [-goal[0],goal[1]]
                            else:
                                new_goal = [goal[0],-goal[1]]
                            human.set_goal_position(new_goal)
                    elif self.train_val_sim == 'circle_static' or self.test_sim == 'circle_static':
                        if i >= self.number_static_humans:
                            # Keep the mobile robots still
                            goal = human.get_goal_position()
                            new_goal = [goal[0], goal[1]]
                            #if goal[0] == self.block_area1[2][0] - human.radius - 1:
                            #    new_goal = [-self.circle_radius,goal[1]]
                            #elif goal[0] == self.block_area2[0][0] + human.radius + 1:
                            #    new_goal = [self.circle_radius, goal[1]]
                            #elif goal[1] == self.block_area1[2][1] - human.radius - 1:
                            #    new_goal = [goal[0], -self.circle_radius]
                            #else:
                            #    new_goal = [goal[0], self.circle_radius]
                            human.set_position(new_goal)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        return ob, reward, done, info, reward_values, left_path, position, is_stopped

    def render(self, mode='human', output_file=None, title=None, test_case=None):
        print('render starts')
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge
        from matplotlib.patches import Rectangle
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        #matplotlib.use("Agg") #download the video
        

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'black'
        goal_color = 'orange'
        arrow_color = 'seagreen'
        arrow_style = patches.ArrowStyle("->", head_length=2, head_width=2)
        arrow_rob_wp = 'gray'

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-20, 20) # GABRIEL map size
            ax.set_ylim(-15, 37)  # GABRIEL map size
            ax.set_aspect('equal')
            ax.set_xlabel('x position (m)', fontsize=16)  # NABIH legend x
            ax.set_ylabel('y position (m)', fontsize=16)  # NABIH legend y
            
            plot_title = ''
            if title is not None:
                plot_title = plot_title + title
                
            if test_case is not None:
                plot_title = plot_title + ' - Test Number: ' + str(test_case)
            ax.set_title(plot_title)
            
            ax.add_artist(Wedge((0.0, 20.6), 13.6, -90, 90, width=6.5, color="lightgray"))
            ax.add_artist(Wedge((0.0, 0.0), 13.6, 90, 270, width=6.5, color="lightgray"))
            ax.add_artist(Rectangle((0.0, -13.6), 13.6, 6.6, facecolor='lightgray'))
            ax.add_artist(Rectangle((-13.6, 27.6), 13.6, 6.6, facecolor='lightgray'))
            
            # Dibujo del path mas corto
            x_sp = []
            y_sp = []
            for i, j in self.short_path:
                x_sp.append(i)
                y_sp.append(j)

            ax.plot(x_sp, y_sp, linewidth = 3, color='magenta')
            
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            robot_goals = [state[0].goal_position for state in self.states]
            #print(len(robot_goals))
            #print(robot_goals)

            # draw a star at the goal position (0,4)
            #goal = mlines.Line2D([4], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot_legend = mlines.Line2D([0], [4], color=robot_color, marker='o', linestyle='None', markersize=15, label='Goal') # NABIH marker
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            goal = mlines.Line2D([-self.big_radius + 4*self.humans[0].radius + 1.3, -self.big_radius + 4*self.humans[0].radius + 1.3], \
                    [self.big_radius + 2*self.small_radius, 2*self.big_radius + self.small_radius], linestyle='dashed', color=goal_color) # X coordinates assume that the radius of the robot and the humans is the same
            goal_legend = mlines.Line2D([0], [4], color=goal_color, linestyle='dashed', markersize=15, label='Goal')
            
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot_legend, goal], ['Robot', 'Goal'], fontsize=16, numpoints=1)   # numpoints=1: only 1 star in the legend
            #plt.legend([robot_legend], ['Robot'], fontsize=16, numpoints=1)   # numpoints=1: only 1 star in the legend

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            human_goals = [[state[1][j].goal_position for j in range(len(self.humans))] for state in self.states]
            #print(human_goals)

            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))]
            #humans_goals = [plt.Circle(human_goals[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='green', fontsize=12) for i in range(len(self.humans))] # nabih human number colors
            for i, human_n in enumerate(human_numbers):
                ax.add_artist(humans[i])
                ax.add_artist(human_n)

            #for human_goal in humans_goals:
            #    ax.add_artist(human_goal)

            
            #Round path
            '''
            circle0 = plt.Circle(self.circle_center, self.inner_circle_radius, fill=False, color='r')
            ax.add_artist(circle0)

            #circle1 = plt.Circle(self.block_area3[0],self.block_area3[1],fill=False, color='r')
            #circle2 = plt.Circle(self.block_area3[0],self.block_area3[2],fill=False, color='r')
            #ax.add_artist(circle1)
            #ax.add_artist(circle2)
            
            outer_circle = []
            #lane_y_coord = 3.5
            #lane_x1_coord = 9.5
            #lane_x2_coord = 22
            #num_elements = 30

            ax.add_artist(plt.Line2D([self.lane_x1_coord, self.lane_x2_coord], [self.lane_y_coord, self.lane_y_coord]))
            ax.add_artist(plt.Line2D([self.lane_x1_coord, self.lane_x2_coord], [-self.lane_y_coord, -self.lane_y_coord]))
            ax.add_artist(plt.Line2D([-self.lane_x1_coord, -self.lane_x2_coord], [self.lane_y_coord, self.lane_y_coord]))
            ax.add_artist(plt.Line2D([-self.lane_x1_coord, -self.lane_x2_coord], [-self.lane_y_coord, -self.lane_y_coord]))

            #Draw of the goal
            ax.add_artist(plt.Line2D([-self.lane_x2_coord+1, -self.lane_x2_coord+1], [-self.lane_y_coord, self.lane_y_coord], linestyle='dashed'))

            #angles = np.linspace(np.arctan(lane_y_coord/lane_x1_coord), np.pi - np.arctan(lane_y_coord/lane_x1_coord), num_elements)
            #outer_circle_radius = np.sqrt(np.power(lane_x1_coord, 2) + np.power(lane_y_coord, 2))
            #circle_center = (0,0)

            for i in range(len(self.angles) - 1):
                x1 = self.outer_circle_radius*np.cos(self.angles[i])
                y1 = self.outer_circle_radius*np.sin(self.angles[i])
                x2 = self.outer_circle_radius*np.cos(self.angles[i+1])
                y2 = self.outer_circle_radius*np.sin(self.angles[i+1])
                ax.add_artist(plt.Line2D([x1, x2], [y1, y2]))

                x3 = -self.outer_circle_radius*np.cos(self.angles[i])
                y3 = -self.outer_circle_radius*np.sin(self.angles[i])
                x4 = -self.outer_circle_radius*np.cos(self.angles[i+1])
                y4 = -self.outer_circle_radius*np.sin(self.angles[i+1])
                ax.add_artist(plt.Line2D([x3, x4], [y3, y4]))
            '''

            #S path
            #small_circle_radius = 7
            #big_circle_radius = 13.3
            angles_lower_circle = np.linspace(self.lower_circle_angles[0], self.lower_circle_angles[1])
            #print('angles')
            #print(angles_lower_circle)

            lower_small_circle =[]
            lower_big_circle = []
            upper_small_circle = []
            upper_big_circle = []

            #min_x = 0
            #temp = 0

            for i in range(len(angles_lower_circle) - 1):
                x1 = self.small_radius*np.cos(angles_lower_circle[i])
                y1 = self.small_radius*np.sin(angles_lower_circle[i])
                x2 = self.small_radius*np.cos(angles_lower_circle[i+1])
                y2 = self.small_radius*np.sin(angles_lower_circle[i+1])
                #lower_small_circle.append([[x1,y1],[x2,y2]])
                ax.add_artist(plt.Line2D([x1,x2], [y1,y2]))

                x3 = self.big_radius*np.cos(angles_lower_circle[i])
                y3 = self.big_radius*np.sin(angles_lower_circle[i])
                x4 = self.big_radius*np.cos(angles_lower_circle[i+1])
                y4 = self.big_radius*np.sin(angles_lower_circle[i+1])
                #lower_big_circle.append([[x3,y3],[x4,y4]])
                ax.add_artist(plt.Line2D([x3,x4], [y3,y4]))
                
                #x5 = (7 + 3.3)*np.cos(angles_lower_circle[i])
                #y5 = (7 + 3.3)*np.sin(angles_lower_circle[i])
                #x6 = (7 + 3.3)*np.cos(angles_lower_circle[i+1])
                #y6 = (7 + 3.3)*np.sin(angles_lower_circle[i+1])
                #lower_big_circle.append([[x3,y3],[x4,y4]])
                #ax.add_artist(plt.Line2D([x5,x6], [y5,y6], linewidth=30, color='gray'))

                #temp = min([x1, x2, x3, x4])
                #min_x = min([temp, min_x])

            #upper_circle_center = (0 + lower_big_circle[0][0][0], self.small_circle_radius + self.big_circle_radius)
            angles_upper_circle = np.linspace(self.upper_circle_angles[0], self.upper_circle_angles[1])

            #max_x = 0
            #temp = 0
            
            for i in range(len(angles_upper_circle) - 1):
                x5 = self.small_radius*np.cos(angles_upper_circle[i]) + self.upper_center[0]
                y5 = self.small_radius*np.sin(angles_upper_circle[i]) + self.upper_center[1]
                x6 = self.small_radius*np.cos(angles_upper_circle[i+1]) + self.upper_center[0]
                y6 = self.small_radius*np.sin(angles_upper_circle[i+1]) + self.upper_center[1]
                #upper_small_circle.append([[x5,y5],[x6,y6]])
                ax.add_artist(plt.Line2D([x5,x6], [y5,y6]))

                x7 = self.big_radius*np.cos(angles_upper_circle[i]) + self.upper_center[0]
                y7 = self.big_radius*np.sin(angles_upper_circle[i]) + self.upper_center[1]
                x8 = self.big_radius*np.cos(angles_upper_circle[i+1]) + self.upper_center[0]
                y8 = self.big_radius*np.sin(angles_upper_circle[i+1]) + self.upper_center[1]
                #upper_big_circle.append([[x7,y7],[x8,y8]])
                ax.add_artist(plt.Line2D([x7,x8], [y7,y8]))

                #temp = max([x5, x6, x7, x8])
                #max_x = max(max_x, temp)

            ax.add_artist(plt.Line2D([self.lower_center[0], self.big_radius], [-self.small_radius, -self.small_radius]))
            ax.add_artist(plt.Line2D([self.lower_center[0], self.big_radius], [-self.big_radius, -self.big_radius]))
            ax.add_artist(plt.Line2D([self.upper_center[0], -self.big_radius], [self.big_radius + 2*self.small_radius, self.big_radius + 2*self.small_radius]))
            ax.add_artist(plt.Line2D([self.upper_center[0], -self.big_radius], [2*self.big_radius + self.small_radius, 2*self.big_radius + self.small_radius]))

            #ax.add_artist(plt.Line2D([lower_small_circle[-1][1][0], max_x], [lower_small_circle[-1][1][1], lower_small_circle[-1][1][1]]))
            #ax.add_artist(plt.Line2D([lower_small_circle[-1][1][0], max_x], [lower_big_circle[-1][1][1], lower_big_circle[-1][1][1]]))
            #ax.add_artist(plt.Line2D([upper_small_circle[-1][1][0], min_x], [upper_small_circle[-1][1][1], upper_small_circle[-1][1][1]]))
            #ax.add_artist(plt.Line2D([upper_small_circle[-1][1][0], min_x], [upper_big_circle[-1][1][1], upper_big_circle[-1][1][1]]))

            #ax.add_artist(plt.Line2D([lower_small_circle[0][0][0], upper_big_circle[0][0][0]], [lower_small_circle[0][0][1], upper_big_circle[0][0][1]]))
            #ax.add_artist(plt.Line2D([lower_big_circle[0][0][0], upper_small_circle[0][0][0]], [lower_big_circle[0][0][1], upper_small_circle[0][0][1]]))


            # add time annotation
            time = plt.text(-1.5, 5, 'Time: {}'.format(0), fontsize=16) #nabih modify text time
            ax.add_artist(time)

            # add distance to way point annotation
            dist_wp = plt.text(-1.5, 2, 'Dist({}) wp: {}'.format(self.dist_orient_wp[0][0], round(self.dist_orient_wp[0][1], 2)), fontsize=16)
            ax.add_artist(dist_wp)

            # add x distance to way point annotation
            dist_x_wp = plt.text(-1.5, -1, 'x: {}'.format(round(self.xs[0], 2)), fontsize=16)
            ax.add_artist(dist_x_wp)

            # add x distance to way point annotation
            dist_y_wp = plt.text(8, -1, 'y: {}'.format(round(self.ys[0], 2)), fontsize=16)
            ax.add_artist(dist_y_wp)

            # compute attention scores
            # if self.attention_weights is not None:
            #     attention_scores = [
            #         plt.text(-11.5, 11 - 1.0 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
            #                  fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            
            arrows = {}
            arrows[0] = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            
            arrows['rob_wp'] = [patches.FancyArrowPatch(*self.rob_wp_vectors[0], color=arrow_rob_wp, arrowstyle=arrow_style, linewidth=2)]
            
            for arrow in arrows[0]:
                ax.add_artist(arrow)

            for arr_wp in arrows['rob_wp']:
                ax.add_artist(arr_wp)

            global_step = {}
            global_step[0] = 0
            ##print(len(self.gif_w_points))
            
            def update(frame_num):
                # nonlocal global_step
                # nonlocal arrows
                #ax.add_artist(Wedge((0.0, 20.6), 13.6, -90, 90, width=6.5, color="lightgray"))
                #ax.add_artist(Wedge((0.0, 0.0), 13.6, 90, 270, width=6.5, color="lightgray"))
                #ax.add_artist(Rectangle((0.0, -13.6), 13.6, 6.6, facecolor='lightgray'))
                #ax.add_artist(Rectangle((-13.6, 27.6), 13.6, 6.6, facecolor='lightgray'))
                
                #humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False) for i in range(len(self.humans))]
                #for human in humans:
                #    ax.add_artist(human)
                
                #robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
                #ax.add_artist(robot)
                #print(frame_num)
                global_step[0] = frame_num
                robot.center = robot_positions[frame_num]
                
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    
                    #Vector to the closest way point
                    for arr_wp in arrows['rob_wp']:
                        arr_wp.remove()
                    
                    arrows['rob_wp'] = [patches.FancyArrowPatch(*self.rob_wp_vectors[frame_num], color=arrow_rob_wp, arrowstyle=arrow_style, linewidth=2)] # for vec in self.rob_wp_vectors]

                    for arr in arrows['rob_wp']:
                        ax.add_artist(arr)

                    # Direction of the robot
                    for arrow in arrows[0]:
                        arrow.remove()
                    
                    arrows[0] = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows[0]:
                        ax.add_artist(arrow)
                    
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        # attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))
            
                human_traj_update = [mlines.Line2D([(human_positions[frame_num][i])[0]], [(human_positions[frame_num][i])[1]], color=cmap(i), marker='.', linestyle='None', markersize=4) for i in range(len(self.humans))]     #nabih human trajectory
                robot_traj_update = [mlines.Line2D([robot.center[0]], [robot.center[1]], color='C4', marker='.', linestyle='None', markersize=4)] # nabih robot trajectory
                for trajec in human_traj_update:
                    ax.add_artist(trajec)           #nabih human trajectory
                for r_trajec in robot_traj_update: 
            
                    ax.add_artist(r_trajec)         # nabih robot trajectory
                
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                dist_wp.set_text('Dist({}) wp: {}'.format(self.dist_orient_wp[frame_num][0], round(self.dist_orient_wp[frame_num][1], 2)))
                dist_x_wp.set_text('x: {}'.format(round(self.xs[frame_num], 2)))
                dist_y_wp.set_text('y: {}'.format(round(self.ys[frame_num], 2)))

                #Draw all way points
                w_p_draw = [mlines.Line2D([value[0]], [value[1]], color='cyan', marker='.', linestyle='None', markersize=7) for value in self.all_w_points]
                for wpd in w_p_draw:
                    ax.add_artist(wpd)
                
                #Draw left way points
                frame_w_points = self.gif_w_points[frame_num]
                w_p_draw = [mlines.Line2D([value[0]], [value[1]], color='black', marker='.', linestyle='None', markersize=7) for value in frame_w_points] 
                for wpd in w_p_draw:
                    ax.add_artist(wpd)

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic', self.robot.kinematics
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step[0] % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))  # z: normalized action values
                z = np.append(z, z[:6])
                z = z.reshape(16, 6)  # rotations: 16   speeds:6
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, cmap=plt.cm.viridis)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            # writer = animation.ImageMagickWriter(
            #     fps=10,
            #     metadata=dict(artist='Gabriel'),
            #     bitrate=1800
            # )

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True
            anim.save('testcase_animation.gif', writer='imagemagick')
        
            if output_file is not None:
                anim1 = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
                anim1.running = True
                #ffmpeg_writer = animation.writers['ffmpeg']
                #writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                #anim1.save(output_file, writer=writer)
                anim1.save(output_file, writer='imagemagick')
            else:
                plt.show()
        else:
            raise NotImplementedError
