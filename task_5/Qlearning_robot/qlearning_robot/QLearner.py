
import numpy as np
import random as rand
from copy import deepcopy


class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2,
                 gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
       
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        # Keep track of the latest state and action which are initialized to 0
        self.s = 0
        self.a = 0

        # Initialize a Q table which records and updates Q value for
        # each action in each state
        self.Q = np.zeros(shape=(num_states, num_actions))
        # Keep track of the number of transitions from s to s_prime for when taking
        # an action a when doing Dyna-Q
        self.T = {}
        # Keep track of reward for each action in each state when doing Dyna-Q
        self.R = np.zeros(shape=(num_states, num_actions))





    def querysetstate(self, s):
       
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        return action



    def author(self):
        return 'harshal'

    def query_set_state(self, s):
       
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.Q[s, :].argmax()
        self.s = s
        self.a = action
        if self.verbose:
            print ("s =", s, "a =", action)
        return action

    def query(self, s_prime, r):
        # Update the Q value of the latest state and action based on s_prime and r
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a]  + self.alpha * (r + self.gamma * self.Q[s_prime, self.Q[s_prime, :].argmax()])

        # Implement Dyna-Q
        if self.dyna > 0:
            # Update the reward table
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r

            if (self.s, self.a) in self.T:
                if s_prime in self.T[(self.s, self.a)]:
                    # count no of occurance
                    self.T[(self.s, self.a)][s_prime] += 1
                else:
                    self.T[(self.s, self.a)][s_prime] = 1
            else:
                self.T[(self.s, self.a)] = {s_prime: 1}

            Q = deepcopy(self.Q)

            for i in range(self.dyna):
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)
                if (s, a) in self.T:
                    # Find the most common s_prime as a result of taking a in s
                    s_pr = max(self.T[(s, a)], key=lambda k: self.T[(s, a)][k])
                    # Update the temporary Q table
                    Q[s, a] = (1 - self.alpha) * Q[s, a]   + self.alpha * (self.R[s, a] + self.gamma
                                              * Q[s_pr, Q[s_pr, :].argmax()])
            # Update the Q table of the learner once Dyna-Q is complete
            self.Q = deepcopy(Q)

        # Find the next action to take and update the latest state and action
        a_prime = self.query_set_state(s_prime)
        self.rar *= self.radr
        if self.verbose:
            print ("s =", s_prime, "a =", a_prime, "r =", r)
        return a_prime





if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
