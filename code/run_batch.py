from agent_bvm import bvmAgent 
from bvm_model import *
from mesa.batchrunner import BatchRunner
from IPython.display import display

if __name__ == '__main__':

    #create model
    fixed_params = {"p": .15, "c": .2, "issues": 3, "l_steps": 100, "n_agents":50}
    variable_params = {"n": range(100,100)}
    batch_run = BatchRunner(
        bvmModel,
        variable_params,
        fixed_params,
        iterations=4,
        model_reporters = {'avg_assort':get_avg_assort}
        )

    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    display(run_data)
