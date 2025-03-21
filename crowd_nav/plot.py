import matplotlib.pyplot as plt
import math


def distance((x1, y1), (x2, y2)):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def plot_circles_and_line(circle_center, circle_radius1, circle_radius2, robot_position,block_area1, block_area2):

    fig, ax = plt.subplots()


    circle1 = plt.Circle(circle_center, circle_radius1, fill=False, color='blue', label='Circle 1')
    circle2 = plt.Circle(circle_center, circle_radius2, fill=False, color='green', label='Circle 2')
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    rect1 = plt.Polygon(block_area1, closed=True, fill=None, edgecolor='red', linestyle='--', label='Rectangle 1')
    rect2 = plt.Polygon(block_area2, closed=True, fill=None, edgecolor='purple', linestyle='--', label='Rectangle 2')
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    dist1 = distance(robot_position, circle_center) - circle_radius1
    dist2 = circle_radius2 - distance(robot_position, circle_center)

    if dist1 < dist2:
	dx = circle_center[0] - robot_position[0]
	dy = circle_center[1] - robot_position[1]
	direction_length = (dx**2 + dy**2)**0.5
	dx /= direction_length
	dy /= direction_length
	x_end = robot_position[0] + dx * dist1
	y_end = robot_position[1] + dy * dist1
    else:
	dx = -circle_center[0] + robot_position[0]
	dy = -circle_center[1] + robot_position[1]
	direction_length = (dx**2 + dy**2)**0.5
	dx /= direction_length
	dy /= direction_length
	x_end = robot_position[0] + dx * dist2
	y_end = robot_position[1] + dy * dist2	

    plt.plot([robot_position[0], x_end], [robot_position[1], y_end], 'b')
    plt.plot(robot_position[0], robot_position[1], 'ro')
    plt.plot(x_end, y_end, 'ro')



    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    plt.legend()
    plt.show()

block = [[-8,-8], 10, 15]
circle_center = block[0]
circle_radius1 = block[1]
circle_radius2 = block[2]
block_area2 = [[-2,5],[-6,5],[-6,2],[-2,2]]
block_area3 = [[6.5,-6],[2.5,-6],[2.5,-9],[6.5,-9]]
        
robot_position = (-4, 4)
print(distance(robot_position, circle_center))
plot_circles_and_line(circle_center, circle_radius1, circle_radius2, robot_position, block_area2, block_area3)



