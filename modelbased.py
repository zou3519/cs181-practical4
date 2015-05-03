import numpy.random as npr
import numpy as np
import csv
import sys

from SwingyMonkey import SwingyMonkey

# parameters
vunit = 20
hunit = 50
gamma = 0.9

class State:

    def __init__(self, hdist_to_trunk, monkey_height, lower_trunk_height):
        self.x = np.floor(hdist_to_trunk*1./hunit)
        self.y = np.floor( (monkey_height -lower_trunk_height)*1. /vunit)

    def __hash__(self):
        return int( (self.x+30)*100+(self.y+20) )

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y 

    def __repr__(self):
        return "(%d,%d)" % (self.x, self.y)
    
class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Q = {} #To store Q(state, action) Q function approximations
        self.P = {} #To store P(state1, state2,action) transition function approximations
        self.R = {} #To store R(state, action) reward function approximations
        self.allStates = [] #To store list of all States

        num_states = ((600.0/hunit)*2 + 1) * ((400.0/vunit)*2 + 1)

        for x in xrange(-600/hunit, 600/hunit):
            for y in xrange(-400/vunit,400/vunit):
                s_xy = State(0,0,0)
                s_xy.x = x
                s_xy.y = y

                #initialize Q function
                self.Q[s_xy, 1] = 0.1 
                self.Q[s_xy, 0] = 0.5

                #initialize reward function: each term is:
                #[Total cumulative reward for (s_xy,action), # total times at (s_xy,action)]
                self.R[s_xy, 1] = [0.1, 1.0]
                self.R[s_xy, 0] = [0.5, 1.0]
                
                self.allStates.append(s_xy)

                for x2 in xrange(-600/hunit, 600/hunit):
                    for y2 in xrange(-400/vunit,400/vunit):
                        s2_xy = State(0,0,0)
                        s2_xy.x = x2
                        s2_xy.y = y2

                        #initialize transition function each term is:
                        #[# of times going s_xy to s2_xy under action, # total times going
                        #from s_xy under action]
                        self.P[s_xy, s2_xy, 1] = [(1.0/num_states),1.0]
                        self.P[s_xy, s2_xy, 0] = [(1.0/num_states),1.0]

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # computes optimal policy and value function value at a given state
    def optimal_future_value(self, state):   
        v1 = self.Q[state,1]
        v2 = self.Q[state,0]
        if v1 > v2:
            return (v1,1)
        else:
            return (v2,0)

    def action_callback(self, state):

        s_i = State(state['tree']['dist'],state['monkey']['bot'], \
            state['tree']['bot'])

        if self.last_state != None:

            #update reward function averages
            self.R[self.last_state,self.last_action][0] = \
                self.R[self.last_state,self.last_action][0] + self.last_reward
            self.R[self.last_state,self.last_action][1] = \
                self.R[self.last_state,self.last_action][1] + 1.0

            #update transition function averages
            for aState in self.allStates:
                self.P[self.last_state,aState,self.last_action][1] = \
                    self.P[self.last_state,aState,self.last_action][1] + 1.0
            self.P[self.last_state,s_i,self.last_action][0] = \
                self.P[self.last_state,s_i,self.last_action][0] + 1.0

            #compute new Q function value for action = 0
            expression = 0.0
            for aState in self.allStates:
                expression = expression + ((self.P[s_i,aState,0][0] / self.P[s_i,aState,0][1]) \
                    * self.optimal_future_value(aState)[0])
            self.Q[s_i,0] = (self.R[s_i,0][0] / self.R[s_i,0][1]) + (gamma * expression)

            #compute new Q function value for action = 1
            expression = 0.0
            for aState in self.allStates:
                expression = expression + ((self.P[s_i,aState,1][0] / self.P[s_i,aState,1][1]) \
                    * self.optimal_future_value(aState)[0])
            self.Q[s_i,1] = (self.R[s_i,1][0] / self.R[s_i,1][1]) + (gamma * expression)

            #pick the better action
            if self.Q[s_i,0] < self.Q[s_i,1]:
                self.last_action = 1
            else:
                self.last_action = 0
                
        else:
            self.last_action = npr.rand() < 0.1

        self.last_state = s_i

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

def writeCSV(scores):
    # Write csv
    resultFile = open("test.csv",'wb')
    wr = csv.writer(resultFile)
    for item in scores:
        wr.writerow([item])

iters = 1000
learner = Learner()

scores = []


for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=1,          # Make game ticks super fast.
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    scores.append(swing.score)
    
    if ii % 5 == 0:
        writeCSV(scores)

    # Reset the state of the learner.
    learner.reset()
writeCSV(scores)







    
