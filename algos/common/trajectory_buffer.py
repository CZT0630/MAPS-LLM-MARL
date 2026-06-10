import numpy as np


class TrajectoryBuffer:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.buffers = [self._new_buffer() for _ in range(num_agents)]

    def _new_buffer(self):
        return {
            'states': [],
            'global_states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'alpha': [],
            'beta': [],
        }

    def add(self, agent_id, state, global_state, action, reward, done, log_prob, value, alpha, beta):
        b = self.buffers[agent_id]
        b['states'].append(state)
        b['global_states'].append(global_state)
        b['actions'].append(action)
        b['rewards'].append(reward)
        b['dones'].append(done)
        b['log_probs'].append(log_prob)
        b['values'].append(value)
        b['alpha'].append(alpha)
        b['beta'].append(beta)

    def compute_advantages(self, gamma=0.99, lam=0.95):
        for b in self.buffers:
            rewards = np.array(b['rewards'], dtype=np.float32)
            values = np.array(b['values'], dtype=np.float32)
            dones = np.array(b['dones'], dtype=np.float32)
            adv = np.zeros_like(rewards)
            last_gae = 0.0
            values_next = np.concatenate([values[1:], np.array([0.0], dtype=np.float32)])
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * (1.0 - dones[t]) * values_next[t] - values[t]
                last_gae = delta + gamma * lam * (1.0 - dones[t]) * last_gae
                adv[t] = last_gae
            returns = adv + values
            b['advantages'] = adv.tolist()
            b['returns'] = returns.tolist()

    def get_batch(self, agent_id):
        b = self.buffers[agent_id]
        return {
            'states': np.array(b['states'], dtype=np.float32),
            'global_states': np.array(b['global_states'], dtype=np.float32),
            'actions': np.array(b['actions'], dtype=np.float32),
            'rewards': np.array(b['rewards'], dtype=np.float32),
            'dones': np.array(b['dones'], dtype=np.float32),
            'log_probs': np.array(b['log_probs'], dtype=np.float32),
            'values': np.array(b['values'], dtype=np.float32),
            'returns': np.array(b['returns'], dtype=np.float32),
            'advantages': np.array(b['advantages'], dtype=np.float32),
            'alpha': np.array(b['alpha'], dtype=np.float32),
            'beta': np.array(b['beta'], dtype=np.float32),
        }

    def clear(self):
        self.buffers = [self._new_buffer() for _ in range(self.num_agents)]