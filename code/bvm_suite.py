from agent_bvm import bvmAgent
from bvm_model import *
from mesa.batchrunner import FixedBatchRunner

class bvmSuite():

    def __init__(self, fixedParams, iters):
        self.batch_run = FixedBatchRunner(
                bvmModel, 
                fixed_parameters = fixedParams,
                iterations=iters,
                model_reporters = {'avg_assort':get_avg_assort, 'opinionClustering':numNonUniformIssues}
                )


    def run(self):
        self.batch_run.run_all()

    def getData(self):
        run_data = self.batch_run.get_model_vars_dataframe()
        return run_data 







