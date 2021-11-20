from bvm_model import *
from model_functions import *
from mesa.batchrunner import FixedBatchRunner
from mesa.batchrunner import BatchRunnerMP
import matplotlib.pyplot as plt
import scipy

class bvmSuite():

    def __init__(self, fixedParams, iters):
        self.N = fixedParams['n_agents']  # Save for later (plot)
        self.batch_run = FixedBatchRunner(
                bvmModel, 
                fixed_parameters = fixedParams,
                iterations=iters,
                model_reporters = {'Steps':getSteps,'Anticlones':getNumAnticlonePairs, 'Clones':getNumClonePairs, 'Buckets':updateBuckets}
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
            
    def plotAvgClone_AntiCloneScatter(self,filename=None):
        if filename!=None:
            data = pd.read_csv(filename)
        else:
            data = self.data
        plt.figure() 
        plt.title('Clones and Anti Clones with o=.20, d=.60')
        plt.xlim(-20,scipy.special.binom(self.N,2)+20)
        plt.ylim(-20,scipy.special.binom(self.N,2)+20)
        plt.scatter(x=data['Clones'],y=data['Anticlones'], alpha=0.5)
        plt.xlabel('Clones')
        plt.ylabel('Anticlones')
        plt.show()
