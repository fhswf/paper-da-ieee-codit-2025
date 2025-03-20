## -------------------------------------------------------------------------------------------------
## -- Paper      : Online-adaptive PID control using Reinforcement Learning
## -- Conference : IEEE International Conference on Control, Decision and Information Technologies (2025)
## -- Authors    : Detlef Arend, Amerik Toni Singh Padda, Andreas Schwung
## -- Development: Detlef Arend, Amerik Toni Singh Padda
## -- Module     : experiment.py
## -------------------------------------------------------------------------------------------------

from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import os
from stable_baselines3 import A2C, PPO, DDPG, SAC
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.systems.pool import PT1,PT2
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import CascadeControlSystem
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.oa.control.controllers import RLPID,wrapper_rl
from mlpro.rl.models import *
from mlpro.rl.models_env import Reward
from mlpro.bf.control import ControlledVariable
from mlpro_int_sb3.wrappers import WrPolicySB32MLPro




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLPIDEnh (RLPID):
    """
    Enhanced RL PID controller.
    """
    
    def __init__(self,
                 p_observation_space, 
                 p_action_space, 
                 p_pid_controller, 
                 p_policy, 
                 p_id = None, 
                 p_buffer_size = 1, 
                 p_ada = True, 
                 p_visualize = False, 
                 p_logging = Log.C_LOG_ALL):


        super().__init__(p_observation_space, 
                         p_action_space, 
                         p_pid_controller, 
                         p_policy, 
                         p_id, 
                         p_buffer_size, 
                         p_ada, 
                         p_visualize, 
                         p_logging)
        

        self._last_error = 0
        self._last_reward = 0
        self._cycle = 0
        self._pid_k = []
        self._pid_tn = []
        self._pid_tv = []
        self._rewards = []
        self._error = []
        self._tstamps=[]


## -------------------------------------------------------------------------------------------------    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:
        """
        Parameters:
        p_sars_elem:SARSElement
            Element of a SARSBuffer
        """

        is_adapted = False

        #get SARS Elements 
        p_state,p_crtl_variable,p_reward,p_state_new = tuple(p_sars_elem.get_data().values())

        #store data 
        self._last_error = p_state_new.get_feature_data().get_values()[0]    
        self._error.append(self._last_error)
        self._rewards.append(self._last_reward)
        kp,tn,tv = tuple(self._pid_controller.get_parameter_values())
        self._pid_k.append(kp)
        self._pid_tn.append(tn)
        self._pid_tv.append(tv)
        self._tstamps.append(self._cycle)
        self._cycle+=1

        
        # start adaptation
        if self._action_old is not None:

           # create a new SARS
            p_sars_elem_new = SARSElement(p_state = p_state,
                                        p_action = self._action_old,
                                        p_reward = p_reward, 
                                        p_state_new = p_state_new)
            
            self._last_reward = p_reward.get_overall_reward()

            #adapt own policy
            is_adapted = self._policy._adapt(p_sars_elem_new)

            if is_adapted:     
                
                # compute new action with new error value (second s of Sars element)
                self._action_old = self._policy.compute_action(p_obs = p_state_new)

                #get the pid paramter values 
                pid_values = self._action_old.get_feature_data().get_values()

                #set paramter pid
                self._pid_controller.set_parameter(p_param={"Kp":pid_values[0],
                                                    "Tn":pid_values[1],
                                                    "Tv":pid_values[2]})           
            
        else:

            #compute new action with new error value (second s of Sars element)
            self._action_old = self._policy.compute_action(p_obs = p_state_new) 

        return is_adapted 


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
# 1. create a custom reward funtion
class MyReward(FctReward):

    def __init__(self, p_logging = Log.C_LOG_NOTHING):
        self._reward = Reward(p_value = 0)
        self._reward_value = 0
        self.error_streak_counter = 0
        super().__init__(p_logging)


## -------------------------------------------------------------------------------------------------   
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable = None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
        reward = - abs(error_new)
        self._reward.set_overall_reward(reward)      
        
        return self._reward

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyReward2(FctReward):

    def __init__(self, p_logging = Log.C_LOG_NOTHING):
        self._reward = Reward(p_value = 0)
        self._reward_value = 0
        self.error_streak_counter = 0
        super().__init__(p_logging)


## -------------------------------------------------------------------------------------------------   
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable = None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
        reward = -abs(error_new)- error_new**2
        self._reward.set_overall_reward(reward)
        
        return self._reward


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyReward3(FctReward):

    def __init__(self, p_logging = Log.C_LOG_NOTHING):
        self._reward = Reward(p_value = 0)
        self._reward_value = 0
        self.error_streak_counter = 0
        super().__init__(p_logging)


## -------------------------------------------------------------------------------------------------   
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable = None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.5
        
        reward = -abs(error_new)- error_new**2 - 10*max(abs(error_new)-e_band,0)**2
        self._reward.set_overall_reward(reward)
           
        return self._reward
    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyReward4(FctReward):

    def __init__(self, p_logging = Log.C_LOG_NOTHING):
        self._reward = Reward(p_value = 0)
        self._reward_value = 0
        self.error_streak_counter = 0
        super().__init__(p_logging)

## -------------------------------------------------------------------------------------------------   
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable = None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.5
        
        reward = -abs(error_new)- error_new**2 - 3*max(abs(error_new)-e_band,0)**2 + 30*min(abs(error_new),0.03)
        self._reward.set_overall_reward(reward)     
        
        return self._reward


## -------------------------------------------------------------------------------------------------   
def experiment_cascade(learning_rate : float, my_reward : FctReward, num_policy : int, num_reward: int, path : str, p_visualize: bool, p_logging: bool):

    # 1 Prepare for test   
    step_rate   = 20
    num_dim     = 1

    # 1.1 Define init parameters and calculate cycle limit

    #init controlled systems parameter
    pt2_K = 1
    pt2_D = 1.6165
    pt2_w_0 = 0.00577
    pt1_T = 1200
    pt1_K = 25

    # define cycle limit
    cycle_limit = 15000

    # init setpoint
    setpoint_value = 40
    status = 'ok'


    # 2 Setup inner casscade

    # 2.1 controlled system 
    my_ctrl_sys_1 = PT1( p_K = pt1_T,
                         p_T = pt1_K,
                         p_sys_num = 0,
                         p_y_start = 0,
                         p_latency = timedelta( seconds = 1 ),
                         p_visualize = p_visualize,
                         p_logging = p_logging )

    my_ctrl_sys_1.reset( p_seed = 42 )   


    # 2.2 P-Controller
    my_ctrl_2 = PIDController( p_input_space = my_ctrl_sys_1.get_state_space(),
                               p_output_space = my_ctrl_sys_1.get_action_space(),
                               p_Kp = 0.36,
                               p_Tn = 0,
                               p_Tv = 0,
                               p_integral_off = True,
                               p_derivitave_off = True,
                               p_name = 'PID Controller2',
                               p_visualize = p_visualize,
                               p_logging = p_logging )


    # 3 Setup outer casscade

    # 3.1 controlled system 
    my_ctrl_sys_2 = PT2( p_K = pt2_K,
                         p_D = pt2_D,
                         p_omega_0 = pt2_w_0,
                         p_sys_num = 1,
                         p_max_cycle = cycle_limit,
                         p_latency = timedelta( seconds = 4 ),
                         p_visualize = p_visualize,
                         p_logging = p_logging )

    my_ctrl_sys_2.reset( p_seed = 42 )


    # 3.2 OAController (Main Controller)

    # 3.2.1 Define PID-Parameter-Space
    p_pid_paramter_space = MSpace() 
    dim_kp = Dimension('Kp',p_boundaries = [0.1,50])
    dim_Tn = Dimension('Tn',p_unit = 'second',p_boundaries = [0,300])
    dim_Tv = Dimension('Tv',p_unit = 'second',p_boundaries = [0,300]) 
    p_pid_paramter_space.add_dim(dim_kp)        
    p_pid_paramter_space.add_dim(dim_Tn)
    p_pid_paramter_space.add_dim(dim_Tv)


    # 3.2.2 Define PID-Output-Space
    p_pid_output_space = MSpace()
    p_control_dim = Dimension('u',p_boundaries = [0,500])
    p_pid_output_space.add_dim(p_control_dim)


    # 3.2.3 Init PID-Controller
    my_ctrl_1 = PIDController( p_input_space = my_ctrl_sys_2.get_state_space(),
                               p_output_space = my_ctrl_sys_2.get_action_space(),
                               p_Kp = 1,
                               p_Tn = 0,
                               p_Tv = 0,
                               p_name = 'PID Controller',
                               p_visualize = p_visualize,
                               p_logging = p_logging )  


    # 3.2.4 Set RL-Policy
    if num_policy == 1:        
        policy_sb3 = A2C( policy="MlpPolicy",learning_rate = learning_rate,seed = 42,env = None,_init_setup_model = False,n_steps = 100)
    elif num_policy == 2:
        policy_sb3 = SAC( policy="MlpPolicy",learning_rate = learning_rate,seed = 42,env = None,_init_setup_model = False,learning_starts = 100)
    elif num_policy == 3:
        policy_sb3 = DDPG( policy="MlpPolicy",learning_rate = learning_rate,seed = 42,env = None,_init_setup_model = False,learning_starts = 100)
    elif num_policy == 4:
        policy_sb3 = PPO( policy="MlpPolicy",learning_rate = learning_rate,seed = 42,env = None,_init_setup_model = False,n_steps = 100)   
    

    # 3.2.5 Init SB3 to MLPro wrapper
    poliy_wrapper = WrPolicySB32MLPro( p_sb3_policy = policy_sb3,
                                       p_cycle_limit = cycle_limit,
                                       p_observation_space = my_ctrl_sys_2.get_state_space(),
                                       p_action_space = p_pid_paramter_space,p_logging = p_logging )


    # 3.2.6 Init PID-Policy
    rl_pid_policy = RLPIDEnh( p_observation_space = my_ctrl_sys_2.get_state_space(),
                              p_action_space = p_pid_output_space,
                              p_pid_controller = my_ctrl_1,
                              p_policy = poliy_wrapper,
                              p_visualize = p_visualize,
                              p_logging = p_logging )


    # 3.2.7 Init OA-PID-Controller
    my_ctrl_OA = wrapper_rl.OAControllerRL( p_input_space = my_ctrl_sys_2.get_state_space(),
                                            p_output_space = p_pid_output_space,
                                            p_rl_policy = rl_pid_policy,
                                            p_rl_fct_reward = my_reward,
                                            p_name = 'RLPID Controller',
                                            p_visualize = p_visualize,
                                            p_logging = p_logging )


    # 4 Cascaded control system
    mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                            p_controllers = [ my_ctrl_OA, my_ctrl_2],
                                            p_controlled_systems = [my_ctrl_sys_2, my_ctrl_sys_1 ],
                                            p_name = 'Stirring vessel',
                                            p_cycle_limit = cycle_limit,
                                            p_visualize = p_visualize,
                                            p_logging = p_logging )


    # 5 Set initial setpoint values for all control workflows (=cascades) of the control system
    for panel_entry in mycontrolsystem.get_control_panels():
        panel_entry[0].set_setpoint( p_values = np.ones(shape = (num_dim)) * setpoint_value )

    

    # 6 Run control cycles
    if p_visualize:
        mycontrolsystem.init_plot( p_plot_settings = PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                                   p_view_autoselect = True,
                                                                   p_step_rate = step_rate,
                                                                   p_plot_horizon = 100 ) )

        input('\n\nPlease arrange all windows and press ENTER to start stream processing...')


    try:
        mycontrolsystem.run()

    except Exception as e:
        print("Fehlermeldung:",e)
        status = e
    

    #store data
    data = {
        "Time stamp": rl_pid_policy._tstamps,
        "Reward": rl_pid_policy._rewards,
        "Error": rl_pid_policy._error,
        "Kp": rl_pid_policy._pid_k,
        "Tn": rl_pid_policy._pid_tn,
        "Tv": rl_pid_policy._pid_tv,
        "setpoint": np.ones(shape = len(rl_pid_policy._tstamps))*setpoint_value,
        "learning_rate": np.ones(shape = len(rl_pid_policy._tstamps))*learning_rate,
        "num_policy": np.ones(shape = len(rl_pid_policy._tstamps))*num_policy,
        "num_reward": np.ones(shape = len(rl_pid_policy._tstamps))*num_reward,
        "Status":[status for i in range(len(rl_pid_policy._tstamps))]}

    # create DataFrame 
    df = pd.DataFrame(data)

    # get current datetime 
    current_date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")  # Format: YYYY-MM-DD  

    # export dataframe as csv
    full_path = os.path.join(path,f"cascaded_control_{current_date}.csv")
    df.to_csv(full_path, index = False)


## ------------------------------------------------------------------------------------------------- 
def get_valid_input(prompt, valid_range, default=None):
    """
    Prompts the user for input with a default value. If the user presses ENTER,
    the default value is used. Input is validated against a given range.
    """
    while True:
        user_input = input(f"{prompt} (Press enter for default: {default}): ")
        if not user_input.strip() and default is not None:
            return default
        try:
            user_input = int(user_input)
            if user_input in valid_range:
                return user_input
            else:
                print(f"Invalid input! Please enter a number between {valid_range[0]} and {valid_range[-1]}.")
        except ValueError:
            print("Invalid input! Please enter a valid number.")


## ------------------------------------------------------------------------------------------------- 
def get_valid_file_path(prompt, default_path):
    """
    Prompts the user for a file path. Uses a default path if the user presses ENTER.
    Validates that the directory exists.
    """
    while True:
        user_input = input(f"{prompt} (Press enter for default: {default_path}): ")
        file_path = user_input.strip() or default_path
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            print(f"Invalid path! The directory '{directory}' does not exist. Please try again.")
        else:
            return file_path




# 1 Demo setup

# 1.1 Default values
policy_range = range(1, 5)  # Valid: 1-4
reward_range = range(0, 4)  # Valid: 0-3
learning_rate_range = range(1, 5)  # Valid: 1-4

default_policy = 1  # Default: A2C
default_reward = 0  # Default: First reward function
default_learning_rate = 1  # Default: 0.001

default_path = os.path.expanduser("~/")

default_visualization = "y"  # Default: Yes
default_logging = "y"  # Default: Yes

# 1.2 Welcome message
print('\n\n--------------------------------------------------------------------------------------------------------------------------')
print('Publication: "Online-adaptive PID control using Reinforcement Learning"')
print('Conference : IEEE International Conference on Control, Decision and Information Technologies (CoDIT) 2025, Split, Croatia')
print('Authors    : Dipl.-Inform. Detlef Arend, M.Sc. Amerik Toni Singh Padda, Prof. Dr.-Ing. Andreas Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : Cascaded control with embedded online-adaptive PID controller')
print('--------------------------------------------------------------------------------------------------------------------------\n')

# 1.3 Safely read the inputs
num_policy = get_valid_input("Please enter the policy number (A2C == 1, SAC == 2, DDPG == 3, PPO == 4)", policy_range, default_policy)
num_reward = get_valid_input("Please enter the reward function (0 - 3)", reward_range, default_reward)
learning_rate = get_valid_input("Please enter the learning rate (0.001 == 1, 0.005 == 2, 0.01 == 3, 0.05 == 4)", learning_rate_range, default_learning_rate)

# 1.4 Get file path with default
file_path = get_valid_file_path("Please enter the file path where the file should be saved", default_path)

# 1.5 Additional user inputs for visualization and logging
visualization = input(f"Enable visualization? (y/n, Press enter for default: {default_visualization}): ").strip().lower() or default_visualization
logging = input(f"Enable logging? (y/n, press Enter for default: {default_logging}): ").strip().lower() or default_logging

# 1.6 Convert inputs to boolean
visualization = visualization == "y"
logging = logging == "y"


# 2 Define mappings
my_rewards = [MyReward(), MyReward2(), MyReward3(), MyReward4()]
learning_rates = [0.001, 0.005, 0.01, 0.05]


# 3 Start control experiment
experiment_cascade(
    learning_rate = learning_rates[learning_rate - 1],
    my_reward = my_rewards[num_reward],
    num_policy = num_policy,
    num_reward = num_reward,
    path = file_path,
    p_visualize = visualization,
    p_logging = logging
)
