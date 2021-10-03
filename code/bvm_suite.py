from agent_bvm import bvmAgent
from bvm_model import *
from mesa.batchrunner import FixedBatchRunner
import matplotlib.pyplot as plt

class bvmSuite():

    def __init__(self, fixedParams, iters):
        self.batch_run = FixedBatchRunner(
                bvmModel, 
                fixed_parameters = fixedParams,
                iterations=iters,
                model_reporters = {'Steps':getSteps,'Anticlones':getMultimodalityStatisticAnticlone, 'Clones':getMultimodalityStatisticClone}
                )
        self.data = None

    def run(self):
        self.batch_run.run_all()
        self.data = self.getData()

    def getData(self):
        run_data = self.batch_run.get_model_vars_dataframe()
        return run_data 

    def plotAvgClone_AntiCloneHist(self,filename=None):
        if filename!=None:
            data = pd.read_csv(filename)
        else:
            data = self.data
        
        data.hist(alpha=0.8, column='Clones', bins=10)
        data.hist(alpha=0.8, column='Anticlones', bins=10)
        plt.show()
        
        

        
        
        



