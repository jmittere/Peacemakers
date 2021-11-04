from bvm_model import *
from mesa.batchrunner import BatchRunner

class bvmSweep():

    def __init__(self, fixedParams,variableParams, iters):
        self.batch_run = BatchRunner(
                bvmModel, 
                fixed_parameters = fixedParams,
                variable_parameters = variableParams,
                iterations=iters,
                model_reporters = {'avg_assort':get_avg_assort}
                )
        self.iterations = iters

    def run(self):
        self.batch_run.run_all()
        self.data = self.getData()

    def getData(self):
        run_data = self.batch_run.get_model_vars_dataframe()
        return run_data 
    
    def plotBucketHeatmap(self,title, filename=None):
        if filename!=None:
            data = pd.read_csv(filename)
        else:
            data = self.data

        plt.figure()
        plt.title(title)
        plt.hist2d(x=data['o'],y=data['d'],weights=data['Buckets'], cmap="viridis", bins=[len(data['o'].unique()),len(data['d'].unique())])
        plt.xlabel("Openness Threshold")
        plt.ylabel("Disgust Threshold")
        plt.xticks(np.arange(0.05, 1, 0.1))
        plt.yticks(np.arange(0.05, 1, 0.1))
        colorBar = plt.colorbar()
        colorBar.set_label('Avg Number of Buckets for Each Suite')
        plt.show()
    
    def plotScatter(self, title, filename=None): 
        if filename!=None:
            data = pd.read_csv(filename)
        else:
            data = self.data

        plt.figure()
        plt.title(title)
        plt.scatter(x=data['o'], y=data['avg_assort'], alpha=.7)
        plt.xticks(np.arange(0.0,1.1,0.1))
        plt.xlabel('Openness Threshold')
        plt.ylabel('Average Assortativity Across All Issues')
        plt.axhline(y=0, linestyle = 'dotted')
        plt.show()

