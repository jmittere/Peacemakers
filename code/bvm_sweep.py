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
                model_reporters = {'Buckets':updateBuckets}
                )
        self.iterations = iters


    def run(self):
        self.batch_run.run_all()
        self.data = self.getData()

    def getData(self):
        run_data = self.batch_run.get_model_vars_dataframe()
        return run_data 
    
    def plotBucketHeatmap(self,filename=None):
        if filename!=None:
            data = pd.read_csv(filename)
        else:
            data = self.data

        plt.hist2d(x=data['o'],y=data['d'],weights=data['Buckets'], cmap="viridis", bins=[len(data['o'].unique()),len(data['d'].unique())])
        plt.xlabel("Openness Threshold")
        plt.ylabel("Disgust Threshold")
        colorBar = plt.colorbar()
        colorBar.set_label('Sum of Number of Buckets for each set of Params')
        plt.show()







