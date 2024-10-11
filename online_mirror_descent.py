from dataclasses import dataclass

import gym
import numpy as np
from gym import spaces
from scipy.special import kl_div

_EPSILON = 10**(-25)


class MirrorDescent:
    """
    Class to compute Online Mirror Descent with changing transition costs
    """

    def __init__(self, env, true_P,reward_type, n_agents=10, lr= 0.01, ):
        self.env = env
        self.lr = lr
        
        # useful as a shortcut
        self.S = env.size * env.size
        self.A = env.action_space.n
        self.N_steps = env.max_steps
        
        # number of agents to observe
        self.n_agents = n_agents

        # state action counts
        self.n_counts = np.zeros((self.S, self.A))
        self.m_counts = np.zeros((self.S, self.A, self.S))

        # initial probability transition
        self.P = np.ones((self.S, self.A, self.S))/self.S

        # initial policy
        self.policy = np.ones((self.N_steps, self.S, self.A))/self.A

        # initial state distribution sequence
        self.mu = np.zeros((self.N_steps, self.S))

        # initial state-action value function
        self.Q = np.zeros((self.N_steps, self.S, self.A))

        # inital algorithm count step
        self.count_step = 0

        # set target
        self.target = self.S - 1

        # count number of couples (s,a) visited
        self.n_state_action_visited = []

        # reward type
        self.reward_type = reward_type

        # true probability kernel
        self.true_P = true_P

        # constrained states
        if self.reward_type == 'constrained':
            self.constrained = self.env.grid.constrained_states()

        # multiple objectives
        if self.reward_type == 'multi_objectives':
            self.multi_objectives = [(self.env.size-1) * self.env.size, self.env.size-1, (self.env.size-1) * self.env.size + self.env.size-1]

        # noise parameters initialization
        self.noise_params = np.zeros(5)


    def nu0(self):
        return self.env.initial_state_dist()

    def reward(self, mu, x):
        """
        mu = vector of size N (\mu(x) for all n \in [N])
        """
        if self.reward_type == 'multi_objectives':
            if x in self.multi_objectives:
                r_mu = 2*(1-mu[:,x])
            else:
                r_mu = 0

        elif self.reward_type == 'constrained':
            if x == ((self.env.size-1) * self.env.size + self.env.size - 1):
                r_mu = 10
            elif x in self.constrained:
                r_mu = -50 *sum(50 * mu[:,state] for state in self.constrained)
            else:
                r_mu = 0

        return r_mu

    def objective_function(self, policy, step):
        mu_dist = self.mu_induced(policy, self.true_P)
        if self.reward_type == 'multi_objectives':
            obj = 0
            for x in self.multi_objectives:
                obj -= (1-mu_dist[step,x])**2
            return -obj

        elif self.reward_type == 'constrained':
            x_target = (self.env.size-1) * self.env.size + self.env.size - 1
            obj = 0
            obj = mu_dist[step, x_target] * 10
            # for x in range(self.S-1):
            #     obj += mu_dist[step,x]
            obj -= np.maximum(0,sum(50 * mu_dist[step,x] for x in self.constrained))**2/2
            return 10-obj


    def softmax(self, y, pi):
        """softmax function
        Args:
          y: vector of len |A|
          pi: vector of len |A|
        """
        max_y = max(y)
        exp_y = [np.exp(self.lr * (y[a] - max_y)) for a in range(y.shape[0])]
        norm_exp = sum(exp_y)
        return [l / norm_exp for l in exp_y]

    def policy_from_logit(self, Q, prev_policy):
        """Compute policy from Q function
        """
        policy = np.zeros((self.N_steps, self.S, self.A))
        for n in range(self.N_steps):
            for x in range(self.S):
                policy[n,x,:] = self.softmax(Q[n,x,:], prev_policy[n,x,:])
                # assert np.sum(policy[n,x,:]) == 1,  'policy should sum to 1'
        
        return policy

    def state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu,x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x]

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] 
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q


    def bonus_state_action_value(self, mu, policy):
        """
        Computes the state-action value function
        (without updating pi)
        """
        Q = np.zeros((self.N_steps, self.S, self.A))

        reward = np.zeros((self.N_steps, self.S))
        for x in range(self.S):
            reward[:,x] = self.reward(mu,x)
            Q[self.N_steps-1,x,:] = reward[self.N_steps-1,x] 

        for n in range(self.N_steps - 1, 0, -1):
            for x in range(self.S):
                for a in range(self.A):
                    Q[n-1,x,a] = reward[n-1,x] + (self.N_steps - n) /np.maximum(np.sqrt(self.n_counts[x,a]),1)
                    for x_next in range(self.S):
                        Q[n-1,x,a] += self.P[x,a,x_next] * np.dot(policy[n, x_next,:], Q[n,x_next,:])

        return Q
    

    def mu_induced(self, policy, P):
        """
        Computes the state distribution induced by a policy
        """
        mu = np.zeros((self.N_steps, self.S))
        mu[0,:] = self.nu0()
        for n in range(1,self.N_steps):
            for x in range(self.S):
                for x_prev in range(self.S):
                    mu[n, x] += mu[n-1, x_prev] * np.dot(policy[n-1, x_prev, :], P[x_prev, :, x])   

        # np.testing.assert_array_equal(np.sum(mu, axis=1), np.ones(self.N_steps), 'proba density should sum to 1')
        return mu 

    
    def iteration(self, n_iterations, bonus):
        """
        """
        if self.count_step == 0:
            self.error_steplast = []
            # define bonus constant
            self.C = np.sqrt(4 * self.S * np.log(self.S * self.A * n_iterations / 0.1))
            for n in range(self.N_steps):
                self.mu[n,:] = self.nu0() 
                self.sum_Q = self.Q
            
        for iter in range(n_iterations):
            print('iteration', iter)
            self.count_step += 1
            # 1b) Update the state-value function
            if bonus == True:
                self.Q = self.bonus_state_action_value(self.mu, self.policy)
            else:
                self.Q = self.state_action_value(self.mu, self.policy)
            # 2b) Compute the policy associated
            self.sum_Q += self.Q
            self.policy = self.policy_from_logit(self.sum_Q, self.policy)
            # 3) Update the probability transitions if needed
            self.P, self.mu_empirical = self.estimate_transition()
            # 4) Update the state-action distribution
            self.mu = self.mu_induced(self.policy, self.P)
            # 5) Compute objective function value at the last time step
            self.error_steplast.append(self.objective_function(self.policy, -1))


    def sample_policy(self, n, state):
        return np.random.choice(self.A, p=self.policy[n, state,:])

    def estimate_transition(self):
        """
        Estimate transitions from n_agents playing the current policy
        """
        n_steps = self.n_agents * self.env.max_steps

        P = np.zeros((self.S, self.A, self.S))
        mu_empirical = np.zeros((self.N_steps, self.S))

        observation = self.env.reset()
        state = self.env.obs_to_state(observation)
        for n in range(n_steps):
            # 1. Sample an action using the policy
            time_step = n % self.env.max_steps
            action = self.sample_policy(time_step, state)
            # 2. Step in the env using this random action
            observation, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.env.obs_to_state(observation)
            # 3. Update state-action counts
            mu_empirical[time_step, state] += 1
            self.n_counts[state, action] += 1
            self.m_counts[state, action, next_state] += 1
            state = next_state.copy()

            if terminated or truncated:
                observation = self.env.reset()
                state = self.env.obs_to_state(observation)

        for s in range(self.S):
            P[:,:,s] = self.m_counts[:,:,s]/np.maximum(1, self.n_counts)

        mu_empirical = mu_empirical/self.n_agents

        n_a_s = np.argwhere(np.sum(P, axis =2) == 0)
        for i in range(len(n_a_s)):
            P[n_a_s[i][0], n_a_s[i][1],:] =  1/self.S 
        self.n_state_action_visited.append(self.S*self.A - len(n_a_s))

        return P, mu_empirical

        # np.testing.assert_array_equal(np.sum(self.P, axis=2), np.ones((self.S, self.A)), 'proba kernel should sum to 1')

    # def estimate_noise(self, t):
    #     """
    #     Estimate noise categorical distribution parameters when we suppose the physical dynamics are known (function g)
    #     t = episode
    #     """
    #     n_steps = self.env.max_steps
    #     noise_traj = np.zeros(5)

    #     observation = self.env.reset()
    #     state = self.env.obs_to_state(observation)
    #     for n in range(n_steps):
    #         # 1. Sample an action using the policy
    #         time_step = n % self.env.max_steps
    #         action = self.sample_policy(time_step, state)
    #         # 2. Step in the env using this random action
    #         observation, reward, terminated, truncated, epsilon = self.env.step(action)
    #         next_state = self.env.obs_to_state(observation)
    #         # 3. Append noise
    #         noise_traj[epsilon] += 1
    #         # 4. Update state
    #         state = next_state.copy()

    #         if terminated or truncated:
    #             observation = self.env.reset()
    #             state = self.env.obs_to_state(observation)

    #     # 4. Update noise parameters
    #     self.noise_params = (n_steps * t * self.noise_params + noise_traj)/(n_steps * (t+1))

    #     # 5. Update probability transition kernel
    #     P = self.env.P(self.noise_params)
    #     return P


        
