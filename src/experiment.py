## -------------------------------------------------------------------------------------------------
## -- Paper      : Online-adaptive PID control using Reinforcement Learning
## -- Conference : IEEE International Conference on Control, Decision and Information Technologies (2025)
## -- Authors    : Detlef Arend, Amerik Toni Singh Padda, Andreas Schwung
## -- Development: Amerik Toni Singh Padda, Detlef Arend
## -- Module     : experiment.py
## -------------------------------------------------------------------------------------------------


from datetime import timedelta, datetime

import numpy as np
from stable_baselines3 import A2C, PPO, DDPG, SAC
import pandas as pd

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.systems.pool import PT1,PT2
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.controlsystems import BasicControlSystem
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

    pass




# 1. create a custom reward funtion
class MyReward(FctReward):

    def __init__(self, p_logging=Log.C_LOG_NOTHING):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
        reward = - abs(error_new)
        self._reward.set_overall_reward(reward)
        #self._reward.set_overall_reward(reward)
        
        return self._reward

# 1. create a custom reward funtion
class MyReward2(FctReward):

    def __init__(self, p_logging=Log.C_LOG_NOTHING):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
        reward = -abs(error_new)- error_new**2
        self._reward.set_overall_reward(reward)
        #self._reward.set_overall_reward(reward)
        
        return self._reward


# 1. create a custom reward funtion
class MyReward3(FctReward):

    def __init__(self, p_logging=Log.C_LOG_NOTHING):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.5
        
        reward = -abs(error_new)- error_new**2 - 10*max(abs(error_new)-e_band,0)**2
        self._reward.set_overall_reward(reward)
        #self._reward.set_overall_reward(reward)
        
        return self._reward
    

# 1. create a custom reward funtion
class MyReward5(FctReward):

    def __init__(self, p_logging=Log.C_LOG_NOTHING):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        e_band = 0.5
        
        reward = -abs(error_new)- error_new**2 - 3*max(abs(error_new)-e_band,0)**2 + 30*min(abs(error_new),0.03)
        self._reward.set_overall_reward(reward)
        #self._reward.set_overall_reward(reward)
        
        return self._reward




def experiment_1(learning_rate,my_reward,num_policy,path,num_reward):

    # 2 Preparation of demo/unit test mode
    pt1_T = 5
    K=5
    t_cycle = pt1_T/(20*K)
    t_sim = 3*pt1_T
    cycle_limit = int(t_sim/t_cycle)

    cycle_limit = cycle_limit*20
    num_dim     = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 2

    setpoint_value = 10



    # 2 Setup the control system

    # 2.1 Controlled system
    my_ctrl_sys = PT1(p_K=K,
                    p_T=pt1_T,
                    p_sys_num=0,
                    p_y_start=0,#setpoint_value,
                    p_latency = timedelta( seconds =t_cycle),
                    p_visualize = visualize,
                    p_logging = logging )
    
    y_max = 50
    my_ctrl_sys.C_BOUNDARIES =[0,y_max]


    # 2.2 Controller
    # Define pid paramter space
    pid_paramter_space = MSpace()
    p_pid_output_space = MSpace()
    p_control_dim = Dimension('u',p_boundaries=[0,int(y_max/5)])
    p_pid_output_space.add_dim(p_control_dim)
    dim_kp = Dimension('Kp',p_boundaries=[0.1,100])
    dim_Tn = Dimension('Tn',p_unit='second',p_boundaries=[0,100])
    dim_Tv= Dimension('Tv',p_unit='second',p_boundaries=[0,100]) 
    pid_paramter_space.add_dim(dim_kp)        
    pid_paramter_space.add_dim(dim_Tn)
    pid_paramter_space.add_dim(dim_Tv)

    # Set a  Policy     
    if num_policy ==1:        
        policy_sb3=A2C( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False) 
    elif num_policy ==2:
        policy_sb3=SAC( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)
    elif num_policy == 3:
        policy_sb3=DDPG( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)
    elif num_policy == 4:
        policy_sb3=PPO( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)


    poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy=policy_sb3,
                                    p_cycle_limit=cycle_limit,
                                    p_observation_space=my_ctrl_sys.get_state_space(),
                                    p_action_space=pid_paramter_space,p_logging= logging)
    



    # create pid controller
    my_pid_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                                 p_output_space = p_pid_output_space,
                                 p_Kp=1,
                                 p_Tn=0,
                                 p_Tv=0,
                                 p_integral_off=False,
                                 p_derivitave_off=False,
                                 p_name = 'PID Controller',
                                 p_visualize = visualize,
                                 p_logging = logging )
    
  

    #create rl pid policy
    rl_pid_policy = RLPIDEnh( p_observation_space=my_ctrl_sys.get_state_space(),
                              p_action_space=p_pid_output_space,
                              p_pid_controller = my_pid_ctrl,
                              p_policy=poliy_wrapper,
                              p_visualize = visualize,
                              p_logging = logging )

    #create OAControllerRL
    my_ctrl = wrapper_rl.OAControllerRL( p_input_space=my_ctrl_sys.get_state_space(),
                                         p_output_space=p_pid_output_space,
                                         p_rl_policy=rl_pid_policy,
                                         p_rl_fct_reward=my_reward,
                                         p_visualize = visualize,
                                         p_logging = logging )


    # 2.3 Basic control system
    mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                          p_controller = my_ctrl,
                                          p_controlled_system = my_ctrl_sys,
                                          p_name = 'First-Order-System',
                                          p_ctrl_var_integration = False,
                                          p_cycle_limit = cycle_limit,
                                          p_visualize = visualize,
                                          p_logging = logging )



    # 3 Set initial setpoint values and reset the controlled system
    mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.ones(shape=(num_dim))*setpoint_value )
    my_ctrl_sys.reset( p_seed = 1 )
    status = "ok"


    # 4 Run some control cycles
    if visualize:
        mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                                p_view_autoselect = True,
                                                                p_step_rate = step_rate,
                                                                p_plot_horizon = 100 ) )
     
    try:
        mycontrolsystem.run()


    except Exception as e:
        print("Fehlermeldung:",e)
        status = e

    
    data = {
        "Zeitstempel": rl_pid_policy._tstamps,
        "Reward": rl_pid_policy._rewards,
        "Error": rl_pid_policy._error,
        "Kp": rl_pid_policy._pid_k,
        "Tn": rl_pid_policy._pid_tn,
        "Tv": rl_pid_policy._pid_tv,
        "setpoint": np.ones(shape=len(rl_pid_policy._tstamps))*setpoint_value,
        "learning_rate": np.ones(shape=len(rl_pid_policy._tstamps))*learning_rate,
        "num_policy": np.ones(shape=len(rl_pid_policy._tstamps))*num_policy,
        "num_reward": np.ones(shape=len(rl_pid_policy._tstamps))*num_reward,
        "Meldung":[status for i in range(len(rl_pid_policy._tstamps))]
    }


    # DataFrame erstellen
    df = pd.DataFrame(data)
    # Aktuelles Datum und Uhrzeit abrufen
    current_date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")  # Format: YYYY-MM-DD  


    # CSV exportieren
    df.to_csv(f"/home/amesi13/Daten_master_arbeit/{path}/experiment_policy_{current_date}.csv", index=False)
    print(f"CSV exportiert als:{path}/experiment_{path}_{current_date}.csv")


def experiment_pt2(learning_rate,my_reward,num_policy,path,num_reward):

    # 1 Preparation of demo/unit test mode
    w_0 = 5
    d = 0.5
    k = 5
    t_cycle = 1/(20*w_0)
    t_sim = 10*1/w_0
    cycle_limit = int(t_sim/t_cycle)

    cycle_limit = cycle_limit*30*10
    num_dim     = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 2

    setpoint_value = 10



    # 2 Setup the control system
    # 2.1 Controlled system
    my_ctrl_sys =   PT2(p_K=k,
                    p_D=d,
                    p_omega_0=w_0,
                    p_sys_num=1,
                    p_max_cycle=cycle_limit,
                    p_latency = timedelta( seconds = t_cycle),
                    p_visualize = visualize,
                    p_logging = logging )
    
    y_max = 50
    my_ctrl_sys.C_BOUNDARIES =[0,y_max]


    # 2.2 Controller
    # Define pid paramter space
    pid_paramter_space = MSpace()
    p_pid_output_space = MSpace()
    p_control_dim = Dimension('u',p_boundaries=[0,int(y_max/5)])
    p_pid_output_space.add_dim(p_control_dim)
    dim_kp = Dimension('Kp',p_boundaries=[0.1,100])
    dim_Tn = Dimension('Tn',p_unit='second',p_boundaries=[0,100])
    dim_Tv= Dimension('Tv',p_unit='second',p_boundaries=[0,100]) 
    pid_paramter_space.add_dim(dim_kp)        
    pid_paramter_space.add_dim(dim_Tn)
    pid_paramter_space.add_dim(dim_Tv)

    # Set a  Policy     
    if num_policy ==1:
        
        policy_sb3=A2C( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)
 
    elif num_policy ==2:
        policy_sb3=SAC( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)
    elif num_policy == 3:
        policy_sb3=DDPG( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)
    elif num_policy == 4:
        policy_sb3=PPO( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False)


    poliy_wrapper = WrPolicySB32MLPro( p_sb3_policy=policy_sb3,
                                       p_cycle_limit=cycle_limit,
                                       p_observation_space=my_ctrl_sys.get_state_space(),
                                       p_action_space=pid_paramter_space,p_logging=logging)
    



    # create pid controller
    my_pid_ctrl = PIDController( p_input_space = my_ctrl_sys.get_state_space(),
                                 p_output_space = p_pid_output_space,
                                 p_Kp=1,
                                 p_Tn=0,
                                 p_Tv=0,
                                 p_integral_off=False,
                                 p_derivitave_off=False,
                                 p_name = 'PID Controller',
                                 p_visualize = visualize,
                                 p_logging = logging )
    
  

    #create rl pid policy
    rl_pid_policy = RLPIDEnh( p_observation_space=my_ctrl_sys.get_state_space(),
                              p_action_space=my_ctrl_sys.get_action_space(),
                              p_pid_controller = my_pid_ctrl,
                              p_policy=poliy_wrapper,
                              p_visualize = visualize,
                              p_logging = logging )

    #create OAControllerRL
    my_ctrl = wrapper_rl.OAControllerRL( p_input_space=MSpace(),
                                         p_output_space=MSpace(),
                                         p_rl_policy=rl_pid_policy,
                                         p_rl_fct_reward=my_reward,
                                         p_visualize = visualize,
                                         p_logging = logging )


    # 2.3 Basic control system
    mycontrolsystem = BasicControlSystem( p_mode = Mode.C_MODE_SIM,
                                          p_controller = my_ctrl,
                                          p_controlled_system = my_ctrl_sys,
                                          p_name = 'First-Order-System',
                                          p_ctrl_var_integration = False,
                                          p_cycle_limit = cycle_limit,
                                          p_visualize = visualize,
                                          p_logging = logging )



    # 3 Set initial setpoint values and reset the controlled system
    mycontrolsystem.get_control_panels()[0][0].set_setpoint( p_values = np.ones(shape=(num_dim))*setpoint_value )
    my_ctrl_sys.reset( p_seed = 1 )
    status="ok"


    #mycontrolsystem.run()
    # 4 Run some control cycles
    if visualize:
        mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                                 p_view_autoselect = True,
                                                                 p_step_rate = step_rate,
                                                                 p_plot_horizon = 100 ) )
     
    try:
        mycontrolsystem.run()


    except Exception as e:
        print("Fehlermeldung:",e)
        status = e

    
    data = {
        "Zeitstempel": rl_pid_policy._tstamps,
        "Reward": rl_pid_policy._rewards,
        "Error": rl_pid_policy._error,
        "Kp": rl_pid_policy._pid_k,
        "Tn": rl_pid_policy._pid_tn,
        "Tv": rl_pid_policy._pid_tv,
        "setpoint": np.ones(shape=len(rl_pid_policy._tstamps))*setpoint_value,
        "learning_rate": np.ones(shape=len(rl_pid_policy._tstamps))*learning_rate,
        "num_policy": np.ones(shape=len(rl_pid_policy._tstamps))*num_policy,
        "num_reward": np.ones(shape=len(rl_pid_policy._tstamps))*num_reward,
        "Meldung":[status for i in range(len(rl_pid_policy._tstamps))]
    }

    # DataFrame erstellen
    df = pd.DataFrame(data)
    # Aktuelles Datum und Uhrzeit abrufen
    current_date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")  # Format: YYYY-MM-DD  


    # CSV exportieren
    df.to_csv(f"/home/amesi13/Daten_master_arbeit/{path}/experiment_policy_{current_date}.csv", index=False)
    print(f"CSV exportiert als:{path}/experiment_{path}_{current_date}.csv")


def experiment_cascade(learning_rate,my_reward,num_policy,path,num_reward):

    # 1 Preparation of demo/unit test mode
    num_dim     = 1
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    step_rate   = 2

    w_0 = 0.00577 
    d = 1.6165
    pt2_K = 1
    pt1_T = 1200
    pt1_K = 25
    t_cycle = pt1_T/(20*pt1_K)
    t_sim = 10*1/w_0
    cycle_limit = 50*int(t_sim/2)
    setpoint_value = 40


    # 2 Setup the control system
    # 2.1 Controller and controlled system of the outer cascade
    my_ctrl_sys_1 = PT1( p_K=pt1_T,
                         p_T=pt1_K,
                         p_sys_num=0,
                         p_y_start=0,
                         p_latency = timedelta( seconds = 1 ),
                         p_visualize = visualize,
                         p_logging = logging )
    
    my_ctrl_sys_1.reset( p_seed = 1 )    

    # 2.2 Controller and controlled system of the inner cascade
    my_ctrl_sys_2 = PT2( p_K=pt2_K,
                         p_D=d,
                         p_omega_0=w_0,
                         p_sys_num=1,
                         p_max_cycle=cycle_limit,
                         p_latency = timedelta( seconds = 4 ),
                         p_visualize = visualize,
                         p_logging = logging )
    
   
    my_ctrl_sys_2.reset( p_seed = 2 )

     # 2.2 Controller
    # Define pid paramter space
    pid_paramter_space = MSpace()
 
    p_pid_output_space = MSpace()
    p_control_dim = Dimension('u',p_boundaries=[0,500])
    p_pid_output_space.add_dim(p_control_dim)
    dim_kp = Dimension('Kp',p_boundaries=[0.1,50])
    dim_Tn = Dimension('Tn',p_unit='second',p_boundaries=[0,300])
    dim_Tv= Dimension('Tv',p_unit='second',p_boundaries=[0,300]) 
    pid_paramter_space.add_dim(dim_kp)        
    pid_paramter_space.add_dim(dim_Tn)
    pid_paramter_space.add_dim(dim_Tv)


    # Set a  Policy
    if num_policy ==1:        
        policy_sb3=A2C( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False,n_steps=100)
    elif num_policy ==2:
        policy_sb3=SAC( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False,learning_starts=100)
    elif num_policy == 3:
        policy_sb3=DDPG( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False,learning_starts=100)
    elif num_policy == 4:
        policy_sb3=PPO( policy="MlpPolicy",learning_rate=learning_rate,seed=42,env=None,_init_setup_model=False,n_steps=100)   


    poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy=policy_sb3,
                                    p_cycle_limit=cycle_limit,
                                    p_observation_space=my_ctrl_sys_2.get_state_space(),
                                    p_action_space=pid_paramter_space,p_logging = logging )


    # 2.2 Controller
    my_ctrl_1 = PIDController( p_input_space = my_ctrl_sys_2.get_state_space(),
                        p_output_space = my_ctrl_sys_2.get_action_space(),
                        p_Kp=1,
                        p_Tn=0,
                        p_Tv=0,
                        p_name = 'PID Controller',
                        p_visualize = visualize,
                        p_logging = logging )   




    #create rl pid policy
    rl_pid_policy = RLPIDEnh(p_observation_space=my_ctrl_sys_2.get_state_space(),
                        p_action_space=p_pid_output_space,
                        p_pid_controller = my_ctrl_1,
                        p_policy=poliy_wrapper,
                        p_visualize = visualize,
                        p_logging = logging )

    #create OAControllerRL
    my_ctrl_OA = wrapper_rl.OAControllerRL(p_input_space=my_ctrl_sys_2.get_state_space()
                                        ,p_output_space=p_pid_output_space
                                        ,p_rl_policy=rl_pid_policy
                                        ,p_rl_fct_reward=my_reward
                                        ,p_visualize = visualize
                                        ,p_logging = logging)
    



    # 2.2 P-Controller
    my_ctrl_2 = PIDController( p_input_space = my_ctrl_sys_1.get_state_space(),
                        p_output_space = my_ctrl_sys_1.get_action_space(),
                        p_Kp=0.36,
                        p_Tn=0,
                        p_Tv=0,
                        p_integral_off=True,
                        p_derivitave_off=True,
                        p_name = 'PID Controller2',
                        p_visualize = visualize,
                        p_logging = logging )
    




    # 2.3 Cascade control system
    mycontrolsystem = CascadeControlSystem( p_mode = Mode.C_MODE_SIM,
                                            p_controllers = [ my_ctrl_OA, my_ctrl_2],
                                            p_controlled_systems = [my_ctrl_sys_2, my_ctrl_sys_1 ],
                                            p_name = 'Stirring vessel',
                                            p_cycle_limit = cycle_limit,
                                            p_visualize = visualize,
                                            p_logging = logging )



    # 3 Set initial setpoint values for all control workflows (=cascades) of the control system
    for panel_entry in mycontrolsystem.get_control_panels():
        panel_entry[0].set_setpoint( p_values = np.ones(shape=(num_dim))* setpoint_value )

    

        # 5 Run some control cycles
    if visualize:
        mycontrolsystem.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                                p_view_autoselect = True,
                                                                p_step_rate = step_rate,
                                                                p_plot_horizon = 100 ) )

    status = 'ok'

    try:
        mycontrolsystem.run()


    except Exception as e:
        print("Fehlermeldung:",e)
        status = e

    
    data = {
        "Zeitstempel": rl_pid_policy._tstamps,
        "Reward": rl_pid_policy._rewards,
        "Error": rl_pid_policy._error,
        "Kp": rl_pid_policy._pid_k,
        "Tn": rl_pid_policy._pid_tn,
        "Tv": rl_pid_policy._pid_tv,
        "setpoint": np.ones(shape=len(rl_pid_policy._tstamps))*setpoint_value,
        "learning_rate": np.ones(shape=len(rl_pid_policy._tstamps))*learning_rate,
        "num_policy": np.ones(shape=len(rl_pid_policy._tstamps))*num_policy,
        "num_reward": np.ones(shape=len(rl_pid_policy._tstamps))*num_reward,
        "Meldung":[status for i in range(len(rl_pid_policy._tstamps))]
    }


    # DataFrame erstellen
    df = pd.DataFrame(data)
    # Aktuelles Datum und Uhrzeit abrufen
    current_date = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")  # Format: YYYY-MM-DD  

    # CSV exportieren
    df.to_csv(f"/home/amesi13/Daten_master_arbeit/{path}/experiment_policy_{current_date}.csv", index=False)
    print(f"CSV exportiert als '{path}/experiment_{path}_{current_date}.csv'")


    
def main (sub_folder,func):

    #init
    policy_numbers = [1,2,3,4]
    my_rewards = [MyReward(),MyReward2(),MyReward3(),MyReward5()]
    learning_rates = [0.001,0.005,0.01,0.05]

    #train loop
    for num_policy in policy_numbers:
        for id_rew,reward in enumerate(my_rewards):
            for learning_rate in learning_rates:         
                func(learning_rate,reward,num_policy=num_policy,path=sub_folder,num_reward= id_rew)




main("pol/PT1",experiment_1)
main("pol/PT2",experiment_pt2)
main("pol/PT1_PT2",experiment_cascade)