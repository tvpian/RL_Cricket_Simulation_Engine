from gym import spaces
import numpy as np


m=3    	#max capacity of warehouse


ball_mapping = {0:'ball1',1:'ball2',2:'ball3',3:'ball4',4:'ball5',5:'ball6'}

class InventoryEnv():

    def __init__(self):
        self.action_space = spaces.Discrete(m)
        self.ball_no = np.random.choice([0,1,2,3,4,])
        self.state = self.ball_no

        # Start the first round
        self.reset()

    def demand(self, ball):
        first = np.random.choice(4)
        second = int((10 - first)*0.5)
        third = 10-(first+second)
    	if ball == 0:
            return [first,second,third]
    	elif ball == 1:
    		return [second,first,third]
    	elif ball == 2:
    		return [second,third,first]
    	elif ball == 3:
    		return [first,second,third]
    	elif ball == 4:
    		return [second,first,third]  
    	else:
    		return [second,third,first]

    def transition(self, x_t_1, a_t_1, d_t):

        if x_t_1 <5:
        	next_ball = x_t_1+1
        else:
        	next_ball = 0    

        
        next_bat_pos=next_ball
        #note that state includes the order which was just delivered
        return (next_bat_pos, next_ball) 

    def reward(self, x_t_1, a_t_1, d_t):
        
        r=score

        return r

    def initial_step(self, state, action):
        assert self.action_space.contains(action)     #to check that action is a discrete value less than m
        obs = state

        if state<5:
            demand = state[1]+1   
        else:
            demand = 0        

        obs2 = self.transition(obs, action, demand)       #next_state

        return obs2

    def step(self, x_t_1, a_t_1):   
        assert self.action_space.contains(a_t_1)     #to check that action is a discrete value less than m
        obs = x_t_1             #at the beginning, state is picked up from the contructor. 
        if x_t_1<5:
            d_t = x_t_1+1    
        else:
            d_t = 0
        obs2 = self.transition(obs, a_t_1, d_t)       #next_state
        reward = self.reward(x_t_1, a_t_1,  d_t)
        return obs2, reward


    def reset(self):
        return self.state
    