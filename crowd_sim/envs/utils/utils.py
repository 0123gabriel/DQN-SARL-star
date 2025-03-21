import numpy as np
import math
from numpy.linalg import norm

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def point_to_arc_dist(cx, cy, radius, start_angle, end_angle, px, py):

    # Calculate the distance from the center to the point
    dist_to_center = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    
    # Project the point in the arc, get the angle between the center and the point
    angle_to_point = math.atan2(py - cy, px - cx)
    
    # Ensure that angles are in the range [0, 2*pi)
    start_angle = start_angle % (2 * math.pi)
    end_angle = end_angle % (2 * math.pi)
    angle_to_point = angle_to_point % (2 * math.pi)
    
    # Auxiliar function to verify that the angle is in the range 
    def is_angle_between(angle, start, end):
        if start < end:
            return start <= angle <= end
        else:
            return start <= angle or angle <= end
    
    # Calculate the closest point on the arc to the given point 
    if is_angle_between(angle_to_point, start_angle, end_angle):

        # If the angle is in the range of the arc
        closest_dist = abs(dist_to_center - radius)
        closest_point = (cx + radius * math.cos(angle_to_point), cy + radius * math.sin(angle_to_point))
    else:
        # If the angle is outside the range of the arc, then calculate the distance to the ends of the arc
        closest_point_start = (cx + radius * math.cos(start_angle), cy + radius * math.sin(start_angle))
        closest_point_end = (cx + radius * math.cos(end_angle), cy + radius * math.sin(end_angle))
        
        dist_to_start = math.sqrt((px - closest_point_start[0]) ** 2 + (py - closest_point_start[1]) ** 2)
        dist_to_end = math.sqrt((px - closest_point_end[0]) ** 2 + (py - closest_point_end[1]) ** 2)
        
        if dist_to_start < dist_to_end:
            closest_dist = dist_to_start
            closest_point = closest_point_start
        else:
            closest_dist = dist_to_end
            closest_point = closest_point_end
    
    #return closest_point, closest_dist
    return closest_dist

def get_agent_direction(far_point, agent_pos):
    vect_dist_wp = np.array(far_point) - np.array(agent_pos)
    return 'h' if abs(vect_dist_wp[0]) > abs(vect_dist_wp[1]) else 'v'

def get_unit_vector(end, start):
    vector = np.array(end) - np.array(start)
    unit_vec = vector/norm(vector)
    return unit_vec

def get_agent_unit_vector(agent):
    radius = agent.radius
    agent_state = agent.get_full_state() 
    if agent.kinematics == 'unicycle':
        orientation = ((agent_state.px, agent_state.py), (agent_state.px + radius*np.cos(agent_state.theta), agent_state.py + radius*np.sin(agent_state.theta)))
            
    else:
        theta = np.arctan2(agent_state.vy, agent_state.vx)
        orientation = ((agent_state.px, agent_state.py), (agent_state.px + radius*np.cos(theta), agent_state.py + radius*np.sin(theta)))

    agent_unit_vec = get_unit_vector(*orientation[::-1])

    return agent_unit_vec

def project_A_on_B(A, B):
    return (np.dot(A, B)/norm(B))*np.array(B)

def get_dir_wp_reward(rob_wp_uvec, rob_dir_uvec):
    proj_vect = project_A_on_B(rob_dir_uvec, rob_wp_uvec)

    if np.dot(rob_wp_uvec, rob_dir_uvec) <= 0:
        return -0.5
    else:
        return np.exp((1/2.0)*norm(proj_vect)) - 1.15


