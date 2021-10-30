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
        
        ci2Data = data.loc[data['CI2'] == True]
        i2Data = data.loc[data['CI2'] == False]
        
        #ci2Data = ci2Data.loc[ci2Data['o']<ci2Data['d']]
        #i2Data = i2Data.loc[i2Data['o']<i2Data['d']]
        #i2Data.to_csv('CleanedI2Data.csv')
        #ci2Data.to_csv('CleanedCI2Data.csv')
        #print(ci2Data)
        #print(i2Data)
        
        plt.figure()
        plt.title("Cross-Issue Influence")
        plt.hist2d(x=ci2Data['o'],y=ci2Data['d'],weights=ci2Data['Buckets'], cmap="viridis", bins=[len(data['o'].unique()),len(data['d'].unique())])
        plt.xlabel("Openness Threshold")
        plt.ylabel("Disgust Threshold")
        colorBar = plt.colorbar()
        colorBar.set_label('Sum of Number of Buckets for each set of Params')
        plt.figure()
        plt.title("Same Issue Influence")
        plt.hist2d(x=i2Data['o'],y=i2Data['d'],weights=i2Data['Buckets'], cmap="viridis", bins=[len(data['o'].unique()),len(data['d'].unique())])
        plt.xlabel("Openness Threshold")
        
        plt.ylabel("Disgust Threshold")
        colorBar = plt.colorbar()
        colorBar.set_label('Sum of Number of Buckets for each set of Params')
        plt.show()







