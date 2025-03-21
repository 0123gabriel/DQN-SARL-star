from copy import copy
import os
import sys
import math
import heapq
import time
#import random  

files_path = '/home/rise2/Gabriel/CrowdNav'
if files_path not in sys.path:
    sys.path.insert(0, files_path)
    
#print(sys.path)

from crowd_sim.envs.utils import env
from crowd_sim.envs.utils.distances import *

class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = env.Env()  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = 10000000
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if math.sqrt((s[0] - self.s_goal[0])**2 + (s[1] - self.s_goal[1])**2) < 0.9:  # stop condition
                if s == self.s_goal: 
                    print('First if')
                    break
                print('Second if')
                self.PARENT[self.s_goal] = s    
                break

            #if s == self.s_goal:  # stop condition
            #    break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = 10000000

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        print(len(self.OPEN))
        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        n = [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]
        n = [(round(value[0], 3), round(value[1], 3))for value in n]
        min_dist = 2.5

        if s[0] > 0 and s[1] < 0:
            for p in copy(n):
                _, dist = point_to_segment_dist_astar(0, -13.6, 13.6, -13.6, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            for p in copy(n):
                _, dist = point_to_segment_dist_astar(0, -7, 13.6, -7, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            for p in copy(n):
                _, dist = point_to_segment_dist_astar(13.6, -7, 13.6, -13.6, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

        if s[0] <= 0 and s[1] < 21:
            cx = 0
            cy = 0
            radius = 7
            start_angle = math.pi/2
            end_angle = -math.pi/2

            for p in copy(n):
                point, dist = point_to_arc_dist_astar(cx, cy, radius, start_angle, end_angle, p[0], p[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            radius_1 = 13.6
            start_angle = math.pi/2
            end_angle = -math.pi/2

            for p in copy(n):
                point, dist = point_to_arc_dist_astar(cx, cy, radius_1, start_angle, end_angle, p[0], p[1])
                if dist < min_dist:
                    n.remove(p)

        if s[0] > 0 and s[1] > 0:
            cx = 0
            cy = 20.6
            radius = 7
            start_angle = -math.pi/2
            end_angle = math.pi/2

            for p in copy(n):
                point, dist = point_to_arc_dist_astar(cx, cy, radius, start_angle, end_angle, p[0], p[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            radius_1 = 13.6
            start_angle = -math.pi/2
            end_angle = math.pi/2

            for p in copy(n):
                point, dist = point_to_arc_dist_astar(cx, cy, radius_1, start_angle, end_angle, p[0], p[1])
                if dist < min_dist:
                    n.remove(p)

        if s[0] < 0 and s[1] >=21:
            for p in copy(n):
                _, dist = point_to_segment_dist_astar(0, 27.6, -13.6, 27.6, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            for p in copy(n):
                _, dist = point_to_segment_dist_astar(0, 34.2, -13.6, 34.2, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

            for p in copy(n):
                _, dist = point_to_segment_dist_astar(-13.6, 27.6, -13.6, 34.2, s[0], s[1])
                if dist < min_dist: #math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2) < radius:
                    n.remove(p)

        return n

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """
        #print('Extract path called')
        path = [self.s_goal]
        s = self.s_goal

        while True:
            #print(s)
            s = PARENT[s]
            path.append(s)

            if s == self.s_start or (abs(s[0] - self.s_start[0]) < 1e-3 and abs(s[1] - self.s_start[1]) < 1e-3):  # stop condition
                #if s == self.s_goal: 
                #    break
                #self.PARENT[self.s_goal] = s    
                break

            #if s == self.s_start:
            #    break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

'''
def main():
    for i in range(15):
        s_start = (round(random.uniform(0, 12), 3), round(random.uniform(-8, -12), 3))
        s_goal = (round(random.uniform(-12, 0), 3), round(random.uniform(29, 33), 3))
        #s_goal = (round(-0.8348198637243396, 2), round(33.38601490094267, 2))
        
        print('\n', i)
        print("S start:", s_start)
        print("S goal:", s_goal)

        

        astar = AStar(s_start, s_goal, "euclidean")
        plot = plotting.Plotting(s_start, s_goal)

        plot.plot_grid_temp("A*")

        path, visited = astar.searching()
        #print(visited)
        plot.animation(path, visited, "A*")  # animation

        # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
        # plot.animation_ara_star(path, visited, "Repeated A*")


if __name__ == '__main__':
    main()
'''
