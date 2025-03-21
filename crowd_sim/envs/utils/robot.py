from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section):
        super(Robot, self).__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        #print('Act robot function: ', list(self.get_full_state()))
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
    
    def step(self, action, m_dist):
        super(Robot, self).step(action)
        self.min_dist = m_dist
        
    def __str__(self):
        return str(list(super(Robot, self).get_full_state()))