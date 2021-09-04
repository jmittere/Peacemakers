from agent_bvm import bvmAgent
from bvm_model import *
from mesa.batchrunner import BatchRunner

class bvmSweep():

    def __init__(self, fixedParams,variableParams, iters):
        self.batch_run = BatchRunner(
                bvmModel, 
                fixed_parameters = fixedParams,
                variable_parameters = variableParams,
                iterations=iters,
                model_reporters = {'avg_assort':get_avg_assort, 'opinionClustering':returnNonUniform}
                )


    def run(self):
        self.batch_run.run_all()

    def getData(self):
        run_data = self.batch_run.get_model_vars_dataframe()
        return run_data 







