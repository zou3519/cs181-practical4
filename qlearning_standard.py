import numpy.random as npr
import numpy as np
import csv
import sys

from SwingyMonkey import SwingyMonkey

# parameters
vunit = 20
hunit = 50
gamma = 0.9
alpha = 0.1
special_init = False

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
        self.Q = {} # map State, Action pairs to real numbers.
        for x in xrange(-600/hunit, 600/hunit):
            for y in xrange(-400/vunit,400/vunit):
                s_xy = State(0,0,0)
                s_xy.x = x
                s_xy.y = y
                if special_init:
                    if y < 0:
                        self.Q[s_xy, 1] = 1.
                    else:
                        self.Q[s_xy, 1] = 0.1
                else:
                    self.Q[s_xy, 1] = 0.1
                self.Q[s_xy, 0] = 0.5

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    # computes max over a of Q_t(s_{t+1}, a)
    def optimal_future_value(self, state):   
        v1 = self.Q[state,1]
        v2 = self.Q[state,0]
        if v1 > v2:
            return (v1,1)
        else:
            return (v2,0)

    def action_callback(self, state):
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.
        s_i = State(state['tree']['dist'],state['monkey']['bot'], \
            state['tree']['bot'])

        if self.last_state != None:
            oldQ = self.Q[self.last_state,self.last_action] 
            (optimal_value, optimal_action) = self.optimal_future_value(s_i)
            self.Q[self.last_state,self.last_action] = oldQ + \
                alpha*(self.last_reward + gamma*optimal_value \
                    - oldQ)

            self.last_action = optimal_action
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

iters = 100
learner = Learner()

scores = []


for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         text="Epoch %d" % (ii), # Display the epoch on screen.
                         tick_length=40,          # Make game ticks super fast.
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







    
