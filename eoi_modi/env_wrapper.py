from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import numpy as np

class env_wrapper(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        super(env_wrapper,self).__init__(world, reset_callback, reward_callback,
                 observation_callback, info_callback,
                 done_callback, shared_viewer)

    def get_state(self):
        state = np.array([])
        for agent in self.agents:
            state = np.append(state, np.concatenate((agent.state.p_pos, agent.state.p_vel, agent.state.c)))
        return state

    def get_obs(self):
        agents_obs = [self._get_obs(agent) for agent in self.agents]
        return agents_obs

    def get_avail_agent_actions(self, agent_id):
        return [1 for i in range(self.action_space[0].n)]
        
    def get_env_info(self):
        env_info = {
            "state_shape": 6*self.n, #每个agent的state都是一个类
            "obs_shape": self.observation_space[0].shape[0],
            "n_actions": self.action_space[0].n,
            "n_agents": self.n,
        }
        return env_info