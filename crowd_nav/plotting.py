from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math


def compute_min_distance_to_block(robot_position, wall_areas):
    min_distance = float('inf')
    max_distance = 0

    for wall_area in wall_areas:
        distance_to_wall, point_on_line = compute_distance_to_wall(robot_position, wall_area)

        if distance_to_wall < min_distance:
            min_distance = distance_to_wall
            closest_point = point_on_line

    return min_distance, closest_point

def compute_distance_to_wall(robot_position, wall_area):
    x, y = robot_position
    distances = {}
    distances_list = []

    for i in range(len(wall_area)):
	
        x1, y1 = wall_area[i]

        x2, y2 = wall_area[(i + 1) % len(wall_area)]  

	distance, point = point_to_line_segment_distance(x, y, x1, y1, x2, y2)
	
	distances[distance] = point
        distances_list.append(distance)

    min_dist = min(distances_list)
    min_point = distances[min_dist]

    return min_dist, min_point

def point_to_line_segment_distance(x, y, x1, y1, x2, y2):
    length_squared = (x2 - x1)**2 + (y2 - y1)**2

    if length_squared == 0:
        return math.sqrt((x - x1)**2 + (y - y1)**2)

    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / length_squared))
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    distance = math.sqrt((x - px)**2 + (y - py)**2)

    return distance, [px, py]

def plot_environment(robot_position, wall_areas):

    plt.scatter(robot_position[0], robot_position[1], color='red', marker='o', label='Robot')


    area_colors = ['blue', 'green', 'orange', 'purple', 'cyan']


    for i, (wall_area, color) in enumerate(zip(wall_areas, area_colors), start=1):
        x_vals = [point[0] for point in wall_area + [wall_area[0]]]
        y_vals = [point[1] for point in wall_area + [wall_area[0]]]
        plt.plot(x_vals, y_vals, color=color, linewidth=2, label='Block Area {}'.format(i))

    min_distance = float('inf')
    closest_point = None
    
    farthest_point, max_distance = farthest_point_and_distance(robot_position, wall_areas)
    plt.plot([robot_position[0], farthest_point[0]], [robot_position[1], farthest_point[1]], linestyle='--', color='yellow', linewidth=1)

    for wall_area in wall_areas:
    	_, closest_point= compute_distance_to_wall(robot_position, wall_area)
    	plt.plot([robot_position[0], closest_point[0]], [robot_position[1], closest_point[1]], linestyle='--', color='black', linewidth=1)


    plt.title('Robot Navigation Environment')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def farthest_point_and_distance(robot_position, wall_areas):
    farthest_point = None
    max_distance = float('-inf')

    for area in wall_areas:
        for vertex in area:
            dist = distance(robot_position, vertex)
            if dist > max_distance:
                max_distance = dist
                farthest_point = vertex

    return farthest_point, max_distance


robot_position = [4, -4]
wall_areas = [
    [[2, 9], [-2, 9], [-2, 5], [2, 5]],
    [[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]],
    [[2, -5], [-2, -5], [-2, -9], [2, -9]],
    [[-5, 2], [-9, 2], [-9, -2], [-5, -2]],
    [[9, 2], [5, 2], [5, -2], [9, -2]]
]

for i in range(100):
    print(robot_position)
    if robot_position[1] < 4: 
	robot_position[1] += 0.1
    elif robot_position[0] > -4:
	robot_position[0] -= 0.1
    else:
	break
    farthest_point, max_distance = farthest_point_and_distance(robot_position, wall_areas)
    print("Max Distance and point:", max_distance, farthest_point)
    min_distance, point= compute_min_distance_to_block(robot_position, wall_areas)
    print("Minimum distance and point:", min_distance - 0.3, point)


plot_environment(robot_position, wall_areas)

# Define the function
def my_function(x):
    return 0.1 * (1 / (1 + np.exp(3*x - 8.5)))**0.1

# Generate x values
x_values = np.linspace(0, 18, 1000)

# Calculate corresponding y values
y_values = my_function(x_values)

# Plot the function
'''plt.plot(x_values, y_values, label='0.1 * (1 / (1 + np.exp(3*x - 8.5)))**0.1')
plt.title('Plot of the Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()'''


