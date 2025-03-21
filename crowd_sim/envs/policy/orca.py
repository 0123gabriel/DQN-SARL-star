from __future__ import division
import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY
import logging
import math


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super(ORCA, self).__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.9
        self.max_speed = 5.56
        self.sim = None

    def configure(self, config):
        # self.time_step = config.getfloat('orca', 'time_step')
        # self.neighbor_dist = config.getfloat('orca', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        # self.radius = config.getfloat('orca', 'radius')
        # self.max_speed = config.getfloat('orca', 'max_speed')
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state
        #print(self_state)
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, params[0],params[1],params[2],params[3], self.radius, self.max_speed)
            self.sim.addAgent(self_state.position, params[0],params[1],params[2],params[3], self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, params[0],params[1],params[2],params[3], human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)

            # @lky add the obstacles
            '''self.sim.addObstacle([[2,9],[-2,9],[-2,5],[2,5]])
            self.sim.addObstacle([[2.5,2.5],[-2.5,2.5],[-2.5,-2.5],[2.5,-2.5]])
            self.sim.addObstacle([[2,-5],[-2,-5],[-2,-9],[2,-9]])
            self.sim.addObstacle([[-5,2],[-9,2],[-9,-2],[-5,-2]])'''

            
            # Curvepath
            '''circle_center = (8, -8)
	    circle_radius1 = 10.0
	    circle_radius2 = 15.0
	
            num_segments = 30  
	    for circle_radius in [circle_radius1, circle_radius2]:
    		for i in range(num_segments):
        	    angle = 2 * math.pi * i / num_segments
        	    next_angle = 2 * math.pi * (i + 1) / num_segments

        	    x1 = circle_center[0] + circle_radius * math.cos(angle)
        	    y1 = circle_center[1] + circle_radius * math.sin(angle)
        	    x2 = circle_center[0] + circle_radius * math.cos(next_angle)
        	    y2 = circle_center[1] + circle_radius * math.sin(next_angle)

        	    self.sim.addObstacle([(x1, y1), (x2, y2)])
            self.sim.processObstacles()'''
            
            #Roundpath
            '''inner_circle = []
            outer_circle = []
            lane_y_coord = 3.5
            lane_x1_coord = 9.5
            lane_x2_coord = 22
            num_elements = 30
            
            self.sim.addObstacle([[lane_x1_coord, lane_y_coord], [lane_x2_coord, lane_y_coord]])
            self.sim.addObstacle([[lane_x1_coord, -lane_y_coord], [lane_x2_coord, -lane_y_coord]])
            self.sim.addObstacle([[-lane_x1_coord, lane_y_coord], [-lane_x2_coord, lane_y_coord]])
            self.sim.addObstacle([[-lane_x1_coord, -lane_y_coord], [-lane_x2_coord, -lane_y_coord]])

            angles = np.linspace(np.arctan(lane_y_coord/lane_x1_coord), np.pi - np.arctan(lane_y_coord/lane_x1_coord), num_elements)
            outer_circle_radius = np.sqrt(np.power(lane_x1_coord, 2) + np.power(lane_y_coord, 2))
            circle_center = (0,0)

            for i in range(len(angles) - 1):
                x1 = outer_circle_radius*np.cos(angles[i])
                y1 = outer_circle_radius*np.sin(angles[i])
                x2 = outer_circle_radius*np.cos(angles[i+1])
                y2 = outer_circle_radius*np.sin(angles[i+1])
                self.sim.addObstacle([(x1,y1), (x2,y2)])

                x3 = -outer_circle_radius*np.cos(angles[i])
                y3 = -outer_circle_radius*np.sin(angles[i])
                x4 = -outer_circle_radius*np.cos(angles[i+1])
                y4 = -outer_circle_radius*np.sin(angles[i+1])
                self.sim.addObstacle([(x3,y3), (x4,y4)])

            inner_circle_radius = 5
            for i in range(num_elements):
                angle = 2 * np.pi * i / num_elements
                next_angle = 2 * np.pi * (i + 1) / num_elements
                x5 = circle_center[0] + inner_circle_radius * np.cos(angle)
                y5 = circle_center[1] + inner_circle_radius * np.sin(angle)
                x6 = circle_center[0] + inner_circle_radius * np.cos(next_angle)
                y6 = circle_center[1] + inner_circle_radius * np.sin(next_angle)
                self.sim.addObstacle([(x5,y5),(x6,y6)])

            self.sim.processObstacles()'''

            #S path
            import matplotlib.pyplot as plt
            small_circle_radius = 7
            big_circle_radius = 13.6
            angles_lower_circle = np.linspace((1.0/2)*np.pi, (3.0/2)*np.pi)

            lower_small_circle =[]
            lower_big_circle = []
            upper_small_circle = []
            upper_big_circle = []

            min_x = 0   
            temp = 0

            for i in range(len(angles_lower_circle) - 1):
                x1 = small_circle_radius*np.cos(angles_lower_circle[i])
                y1 = small_circle_radius*np.sin(angles_lower_circle[i])
                x2 = small_circle_radius*np.cos(angles_lower_circle[i+1])
                y2 = small_circle_radius*np.sin(angles_lower_circle[i+1])
                lower_small_circle.append([[x1,y1],[x2,y2]])
                #plt.plot([x1,x2], [y1,y2])
                self.sim.addObstacle([(x1,y1),(x2,y2)])

                x3 = big_circle_radius*np.cos(angles_lower_circle[i])
                y3 = big_circle_radius*np.sin(angles_lower_circle[i])
                x4 = big_circle_radius*np.cos(angles_lower_circle[i+1])
                y4 = big_circle_radius*np.sin(angles_lower_circle[i+1])
                lower_big_circle.append([[x3,y3],[x4,y4]])
                #plt.plot([x3,x4], [y3,y4])
                self.sim.addObstacle([(x3,y3),(x4,y4)])

                temp = min([x1, x2, x3, x4])
                min_x = min([temp, min_x])

            upper_circle_center = (0 + lower_big_circle[0][0][0],small_circle_radius + big_circle_radius)
            angles_upper_circle = np.linspace(-(1.0/2)*np.pi, (1.0/2)*np.pi)

            max_x = 0
            temp = 0

            for i in range(len(angles_upper_circle) - 1):
                x5 = small_circle_radius*np.cos(angles_upper_circle[i]) + upper_circle_center[0]
                y5 = small_circle_radius*np.sin(angles_upper_circle[i]) + upper_circle_center[1]
                x6 = small_circle_radius*np.cos(angles_upper_circle[i+1]) + upper_circle_center[0]
                y6 = small_circle_radius*np.sin(angles_upper_circle[i+1]) + upper_circle_center[1]
                upper_small_circle.append([[x5,y5],[x6,y6]])
                #plt.plot([x5,x6], [y5,y6])
                self.sim.addObstacle([(x5,y5),(x6,y6)])

                x7 = big_circle_radius*np.cos(angles_upper_circle[i]) + upper_circle_center[0]
                y7 = big_circle_radius*np.sin(angles_upper_circle[i]) + upper_circle_center[1]
                x8 = big_circle_radius*np.cos(angles_upper_circle[i+1]) + upper_circle_center[0]
                y8 = big_circle_radius*np.sin(angles_upper_circle[i+1]) + upper_circle_center[1]
                upper_big_circle.append([[x7,y7],[x8,y8]])
                #plt.plot([x7,x8], [y7,y8])
                #plt.axis('equal')
                self.sim.addObstacle([(x7,y7),(x8,y8)])

                temp = max([x5, x6, x7, x8])
                max_x = max(max_x, temp)
            
            self.sim.addObstacle([[lower_small_circle[-1][1][0], lower_small_circle[-1][1][1]], [max_x, lower_small_circle[-1][1][1]]])
            self.sim.addObstacle([[lower_small_circle[-1][1][0], lower_big_circle[-1][1][1]], [max_x, lower_big_circle[-1][1][1]]])
            self.sim.addObstacle([[upper_small_circle[-1][1][0], upper_small_circle[-1][1][1]], [min_x, upper_small_circle[-1][1][1]]])
            self.sim.addObstacle([[upper_small_circle[-1][1][0], upper_big_circle[-1][1][1]], [min_x, upper_big_circle[-1][1][1]]])

            self.sim.addObstacle([[lower_small_circle[0][0][0], lower_small_circle[0][0][1]], [upper_big_circle[0][0][0], upper_big_circle[0][0][1]]])
            self.sim.addObstacle([[lower_big_circle[0][0][0], lower_big_circle[0][0][1]], [upper_small_circle[0][0][0], upper_small_circle[0][0][1]]])

            self.sim.processObstacles()

        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        human_vmax = 5.56
        if speed != 0:
            # human_vmax = np.random.uniform(0, 1.5)
            pref_vel = human_vmax * (velocity / speed)
        else:
            pref_vel = np.array([0,0])

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action
