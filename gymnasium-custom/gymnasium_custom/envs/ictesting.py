import gymnasium as gym # 0.27.1
from gymnasium import logger, spaces
import pygame # 2.3.0
import numpy as np # 1.22.4
import math
from gymnasium.error import DependencyNotInstalled
import pandas as pd
import pyreadr as pr
from dotenv import dotenv_values
from os import path


class IcTestEnvironment(gym.Env):
    """
    Description ...
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, data=False):
        
        if data:
            self.data_path = data
        else:
            env_variables = dotenv_values('.env')
            file_name = env_variables['TRAINING_FILE']
            file_path = env_variables['DATA_PATH']
            self.data_path = path.join(file_path, file_name)
            
        self.data = pd.read_csv(self.data_path, delimiter = ";").to_numpy() # read .csv file, convert pd.df to np.array
        self.test_data = self.data[0:, 8:] # slice array to relevant test data
        self.cond_label = self.data[0:,0] # slice array to IC-condition labels (1: good device, other: bad device)
        
        
        self.no_of_tests = np.shape(self.test_data)[1] # ammount of tests in data
        self.no_of_duts = len(self.cond_label) # ammount of DUTs in data
        self.test_no = 0 # initial Test No.
        self.dut_id = -1 # initial DUT ID
        
        self.dut_cond = None # Agent based IC Condition, True: good device, False: bad device
        self.true_dut_cond = None # True IC Condition, True: good device, False: bad device
        
        # State space: [Test No., Test Result]
        # low: -Inf
        states_low = [-np.finfo(np.float32).max] * self.no_of_tests
        low = np.array(states_low, dtype=np.float32)
        # high: Inf
        states_high = [np.finfo(np.float32).max] * self.no_of_tests
        high = np.array(states_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.state = None
        
        # 3 actions corresponding to 0: "abort good", 1: "abort bad", 2: "continue"
        self.action_space = spaces.Discrete(3)
        
        # define instant reward (to be used in reward functions)
        self.inst_reward = 0
        
        # In case of process visualization:
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
               
        # Ammount of steps after a terminating state
        self.steps_beyond_terminated = None
        
        
    def step(self, action):
        """
        This method defines both reward and next state in dependence of an action taken by the agent.
        Returns: the next state, the reward, whether the episode is terminated or not and optionally additional info
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method." 
        
        self.test_no +=1
        
        if action == 2:
            test_no = float(self.test_no)
            test_result = self.test_data[self.dut_id, self.test_no]
            
        elif action == 1:
            test_no = float(self.test_no)
            test_result = 0 
            self.dut_cond = False # bad device
            
        elif action == 0:
            test_no = float(self.test_no)
            test_result = 0 
            self.dut_cond = True # good device
        
        # overwrite state
        self.state[self.test_no] = test_result
        
        overdue = (self.test_no == (self.no_of_tests-1))

        terminated = bool(action == 1 or action == 0 or overdue)
        
        # Reward Functions:
        #tbr = np.tanh(2e-3*self.test_no) - 8e-4 * self.test_no # tbr V1
        tbr = -1e-1 # tbr V2

        if not terminated:
            reward = tbr
        
        elif self.steps_beyond_terminated is None:
            
            self.steps_beyond_terminated = 0

            if self.cond_label[self.dut_id] == 1:
                self.true_dut_cond = True

            elif self.cond_label[self.dut_id] != 1:
                self.true_dut_cond = False

            print("DUT No.: {}".format(self.dut_id), end = '\r', flush = True)

            if self.dut_cond == None:
                reward = -800
                
            elif self.dut_cond == False and not self.true_dut_cond:
                reward = 250*np.tanh(2e-3*self.test_no - 1.5) + 250
                
            elif self.dut_cond == True and self.true_dut_cond:
                reward = 250*np.tanh(2e-3*self.test_no - 1.5) + 250
                
            else:
                reward = -300 * np.tanh(2e-3*self.test_no - 1.5) - 400
        
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
                
            self.steps_beyond_terminated += 1
            reward = 0.0
            
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {'PC' : self.dut_cond, 'TC' : self.true_dut_cond}
    
    
    def reset(self, seed=None, options=None):
        """
        Resets to the initial state (Test 0, Test Result 0, pending test results defined as infinite).
        Calling the method indicates, that the testing process proceeds by testing the next DUT.
        Returns the initial state.
        """

        # call reset of parent class Env in core.py
        super().reset(seed=seed) 
        
        # reset attributes 
        self.dut_cond = None # set condition to unknown
        self.true_dut_cond = None # set true condition to unknown
        
        if self.dut_id == (self.no_of_duts-1): # Out of data (end of epoch), reset to DUT 0
            self.dut_id = 0
        else:
            self.dut_id += 1 # next DUT
            
        self.test_no = 0 # reset to test 0
        
        # set intial state: test result 0 + rest of list filled with value 1 * (no_of_tests - 1) 
        self.state = [self.test_data[self.dut_id, 0]] + [1] * (self.no_of_tests-1)
        self.steps_beyond_terminated = None
        
        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        """
        Mandatory, yet unused in the case of IC testing.
        """
        pass