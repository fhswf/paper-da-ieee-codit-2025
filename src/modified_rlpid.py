## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.oa.control.controllers
## -- Module  : oa_pid_controller.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-09-01  0.0.0     DA       Creation 
## -- 2024-09-26  0.1.0     ASP      Implementation RLPID, RLPIDOffPolicy 
## -- 2024-10-17  0.2.0     ASP      -Refactoring class RLPID
## --                                -change class name RLPIDOffPolicy to OffPolicyRLPID
## -- 2024-11-10  0.3.0     ASP      -Removed class OffPolicyRLPID
## -- 2024-12-05  0.4.0     ASP      -Add plot methods
## -- 2024-12-05  0.5.0     ASP      -changed signature of compute_action()
## -- 2024-12-05  0.6.0     ASP      -implementation assign_so(), update compute_action()
## -- 2024-12-06  0.7.0     ASP      -BugFix: _adapt()
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.0 (2024-12-05)

This module provides an implementation of a OA PID controller.

"""

from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.ml.basics import *
from mlpro.rl import Policy,SARSElement
from mlpro.rl import Action, State, SARSElement, FctReward, Policy
from mlpro.bf.streams import Instance,InstDict
import numpy as np




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLPID(Policy):
    """
    Policy class for closed loop control

    Parameters
    ----------
    p_pid_controller : PIDController,
        Instance of PIDController
    p_policy : Policy
        Policy algorithm
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                  p_observation_space: MSpace,
                  p_action_space: MSpace,
                  p_pid_controller:PIDController ,
                  p_policy:Policy,
                  p_id=None, 
                  p_buffer_size: int = 1, 
                  p_ada: bool = True, 
                  p_visualize: bool = False,
                  p_logging=Log.C_LOG_ALL ):
        
        super().__init__(p_observation_space, p_action_space, p_id, p_buffer_size, p_ada, p_visualize, p_logging)

        self._pid_controller = p_pid_controller
        self._policy = p_policy
        self._action_old = None #None
        self._action_space = p_action_space

    
        self._max_plots =5
        self._plot_nd_plots = None 
        self._plot_nd_xdata =[]
        self._plot_nd_ymin =np.zeros(self._max_plots)
        self._plot_nd_ymax =np.zeros(self._max_plots)
        self._plot_labels = ["$u_t$", "$r_t$", "$K_{p,t}$","$T_{n,t}$ [s]","$T_{v,t}$ [s]"]
        self._last_reward = 0


        #löschen
        self._last_error=0
        self._cycle =0
        self._pid_k = []
        self._pid_tn = []
        self._pid_tv = []
        self._rewards = []
        self._error = []
        self._tstamps=[]


        #ende löschen 



## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 Create a dispatcher hyperparameter tuple for the RLPID policy
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

        # 2 Extend RLPID policy's hp space and tuple from policy
        try:
            self._hyperparam_space.append( self._policy.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._policy.get_hyperparam())
        except:
            pass

## -------------------------------------------------------------------------------------------------
    def get_hyperparam(self) -> HyperParamTuple:
       return self._policy.get_hyperparam()
    
    
## -------------------------------------------------------------------------------------------------
    def _update_hyperparameters(self) -> bool:
       return self._policy._update_hyperparameters()  
      

## -------------------------------------------------------------------------------------------------    
    def _adapt(self, p_sars_elem: SARSElement) -> bool:
        """
        Parameters:
        p_sars_elem:SARSElement
            Element of a SARSBuffer
        """

        is_adapted = False

        #get SARS Elements 
        p_state,p_crtl_variable,p_reward,p_state_new=tuple(p_sars_elem.get_data().values())

        #löschen nach Training
        self._last_error = p_state_new.get_feature_data().get_values()[0]    
        self._error.append(self._last_error)
        self._rewards.append(self._last_reward)
        kp,tn,tv = tuple(self._pid_controller.get_parameter_values())
        self._pid_k.append(kp)

        self._pid_tn.append(tn)
        self._pid_tv.append(tv)
        self._tstamps.append(self._cycle)
        self._cycle+=1
        #ende löschen


        
        if self._action_old is not None:



           # create a new SARS
            p_sars_elem_new = SARSElement(p_state=p_state,
                                        p_action=self._action_old,
                                        p_reward=p_reward, 
                                        p_state_new=p_state_new)
            
            self._last_reward = p_reward.get_overall_reward()

            #adapt own policy
            is_adapted = self._policy._adapt(p_sars_elem_new)

            if is_adapted:     
                
                # compute new action with new error value (second s of Sars element)
                self._action_old=self._policy.compute_action(p_obs=p_state_new)

                #get the pid paramter values 
                pid_values = self._action_old.get_feature_data().get_values()

                #set paramter pid
                self._pid_controller.set_parameter(p_param={"Kp":pid_values[0],
                                                    "Tn":pid_values[1],
                                                    "Tv":pid_values[2]})
            
            
        else:

            #compute new action with new error value (second s of Sars element)
            self._action_old = self._policy.compute_action(p_obs=p_state_new) 

        return is_adapted 
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs: State) -> Action:  

        #get action 
        control_variable=self._pid_controller.compute_output(p_ctrl_error=p_obs)

        #return action
        return Action(p_action_space=control_variable.get_feature_data().get_related_set(),
               p_values=control_variable.values, 
               p_tstamp=control_variable.tstamp)       
    
    
## -------------------------------------------------------------------------------------------------
    def _init_plot_2d(self, p_figure, p_settings):
        return self._pid_controller._init_plot_2d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_3d(self, p_figure, p_settings):
        return self._pid_controller._init_plot_3d(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _init_plot_nd(self, p_figure, p_settings):

        p_settings.axes = p_figure.subplots(self._max_plots,sharex=True)
        # Gemeinsame X-Achsenbeschriftung
        p_figure.text(0.5, 0.04, "t [s]", ha='center')
        p_figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

        #return self._pid_controller._init_plot_nd(p_figure, p_settings)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_2d(self, p_settings, p_inst, **p_kwargs):
        return self._pid_controller._update_plot_2d(p_settings, p_inst, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_3d(self, p_settings, p_inst, **p_kwargs):
        return self._pid_controller._update_plot_3d(p_settings, p_inst, **p_kwargs)
    

## -------------------------------------------------------------------------------------------------
    def _update_plot_nd(self, p_settings, p_inst:InstDict, **p_kwargs): 

        #return self._pid_controller._update_plot_nd(p_settings, p_inst, **p_kwargs)
    
        
        y_valaues =[]  
        
        # 1 Check: something to do?
        if len(p_inst) == 0: return

        #init
        if self._plot_nd_plots is None:
            
            self._plot_nd_plots = {}

            for idx,ax in enumerate(p_settings.axes):
                #control Variable
                if idx == 0:                    
                    inst_ref = next(iter(p_inst.values()))[1]
                    # 2.1 Add plot for each feature
                    
                    feature_space = inst_ref.get_feature_data().get_related_set()

                    self._plot_nd_plots[idx] =[]
                    for feature in feature_space.get_dims():
                        if feature.get_base_set() in [ Dimension.C_BASE_SET_R, Dimension.C_BASE_SET_N, Dimension.C_BASE_SET_Z ]:

                            feature_xdata = []
                            feature_ydata = []
                            feature_plot, = ax.plot( feature_xdata, 
                                                                feature_ydata, 
                                                                lw=1 )

                            self._plot_nd_plots[idx].append( [feature_ydata, feature_plot] )

 
                else:        
                    self._plot_nd_plots[idx] =[]
                    feature_plot, = ax.plot( [],[],lw=1 )
                    self._plot_nd_plots[idx].append( [[], feature_plot] )

            

        # 3 Update plot data
        for inst_id, (inst_type, inst) in sorted(p_inst.items()):

            try:
                self._plot_nd_xdata.append(inst.tstamp.total_seconds())
            except:
                self._plot_nd_xdata.append(inst.tstamp)

            feature_data = inst.get_feature_data().get_values()                    
               


            # 5 Set new plot data of all feature plots
            for i,fplot in enumerate(self._plot_nd_plots[0]):
                fplot[0].append(feature_data[i])
                y_valaues.append(feature_data[i]) 
                fplot[1].set_xdata(self._plot_nd_xdata)
                fplot[1].set_ydata(fplot[0])
                fplot[1].set_label("Control Variable")

        kp,tn,tv = tuple(self._pid_controller.get_parameter_values())
        for key in self._plot_nd_plots.keys():
            #Control Variable
            if key == 0: continue

            # Reward
            elif key == 1:         
                for i, fplot in enumerate(self._plot_nd_plots[key]):               
                    fplot[0].append(self._last_reward)
                    fplot[1].set_xdata(self._plot_nd_xdata)
                    fplot[1].set_ydata(fplot[0])
                    fplot[1].set_label("$Reward$") 
                     
                    y_valaues.append(self._last_reward)

            #K_p
            elif key == 2:
                 for i, fplot in enumerate(self._plot_nd_plots[key]):              
                    fplot[0].append(kp)
                    fplot[1].set_xdata(self._plot_nd_xdata)
                    fplot[1].set_ydata(fplot[0])
                    fplot[1].set_label("$K_p$")
                    y_valaues.append(kp)

            #Tn
            elif key == 3:
                 for i, fplot in enumerate(self._plot_nd_plots[key]):              
                    fplot[0].append(tn)
                    fplot[1].set_xdata(self._plot_nd_xdata)
                    fplot[1].set_ydata(fplot[0])
                    fplot[1].set_label("$T_n$")
                    y_valaues.append(tn)
            
            #Tv
            elif key == 4:
                 for i, fplot in enumerate(self._plot_nd_plots[key]):              
                    fplot[0].append(tv)
                    fplot[1].set_xdata(self._plot_nd_xdata)
                    fplot[1].set_ydata(fplot[0])
                    fplot[1].set_label("$T_v$")
                    y_valaues.append(tv)



        # check y_limits
        for idx,y_value in enumerate(y_valaues):
            if ( self._plot_nd_ymin[idx] is None ) or ( self._plot_nd_ymin[idx] > y_value ):
                self._plot_nd_ymin[idx] = y_value

            #if ( self._plot_nd_ymax[idx] is None ) or  (self._plot_nd_ymax[idx] < y_value ):
            self._plot_nd_ymax[idx] = y_value +0.5        






        for idx,ax in enumerate(p_settings.axes):
            ax.set_ylabel(self._plot_labels[idx])
            ax.legend()
            ax.set_ylim(self._plot_nd_ymin[idx],self._plot_nd_ymax[idx])
        # 6 Update ax limits
            if p_settings.plot_horizon > 0:
                xlim_id = max(0, len(self._plot_nd_xdata) - p_settings.plot_horizon)
            else:
                xlim_id = 0

            if isinstance(self._plot_nd_xdata[xlim_id], timedelta):
                # Handling if the tstamps are timedeltas
                try:
                    ax.set_xlim(self._plot_nd_xdata[xlim_id].total_seconds(), self._plot_nd_xdata[-1].total_seconds())
                except:
                    raise Error("time delta could not be processed")
            else:
                ax.set_xlim(self._plot_nd_xdata[xlim_id], self._plot_nd_xdata[-1])
        #p_settings.axes.set_ylim(self._plot_nd_ymin, self._plot_nd_ymax)    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_2d(self):
        return self._pid_controller._remove_plot_2d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_3d(self):
        return self._pid_controller._remove_plot_3d()
    

## -------------------------------------------------------------------------------------------------
    def _remove_plot_nd(self):
        return self._pid_controller._remove_plot_nd()


## -------------------------------------------------------------------------------------------------
    def assign_so(self, p_so:Shared):
        """
        Assigns an existing shared object to the task. The task takes over the range of asynchronicity
        of the shared object if it is less than the current one of the task.

        Parameters
        ----------
        p_so : Shared
            Shared object.
        """

        self._pid_controller.assign_so(p_so=p_so)