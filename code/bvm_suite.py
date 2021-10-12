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
        plt.figure() 
        plt.hist(data['Clones'], alpha=0.5, label='Clones')
        plt.hist(data['Anticlones'], alpha=0.5, label='Anticlones')
        plt.ylabel('Frequencies')
        plt.legend(loc='best')
        plt.show()
        
        

        
        
        



