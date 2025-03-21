class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'ReachingGoal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'TooClose'


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'
    
class CollisionWall(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'CollisionWall'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''

class Loser(object):
    def __init__(self):
        pass

    def __str__(self):
        return'Another_robot_won'
