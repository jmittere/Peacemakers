from agent_bvm import bvmAgent 
from bvm_model import *
from mesa.batchrunner import BatchRunner

def plotScatter(df, var1, var2, var1label = "Variable X", var2label = "Variable Y"):
    x = df[var1]
    y = df[var2]

    plt.scatter(x,y)

    if(var1label == "Variable X"):
        plt.xlabel("{}".format(var1))
        plt.ylabel("{}".format(var2))
    else:
        plt.xlabel("{}".format(var1label))
        plt.ylabel("{}".format(var2label))
    plt.xticks(np.arange(50,56,1))
    #plt.yticks(np.arange(0,1.1,.1)))
    #plt.title('Log of y axis Plot for H1A')
    plt.show()

def plotHeat(df, var1, var2, var1label = "Variable X", var2label = "Variable Y"):

    plt.hist2d(x = df[var1], y = df[var2], cmap='viridis', bins = [len(df[var1].unique()),4]) #second bin dimension needs to be changed depending on what we are graphing
    colorBar = plt.colorbar()
    colorBar.set_label('Number of simulation runs')

    if(var1label == "Variable X"):
        plt.xlabel("{}".format(var1))
        plt.ylabel("{}".format(var2))
    else:
        plt.xlabel("{}".format(var1label))
        plt.ylabel("{}".format(var2label))
    
    #plt.xticks(np.arange(50,56,1))
    #plt.yticks(np.arange(0,1.1,.1)))
    #plt.title('Log of y axis Plot for H1A')
    plt.show()


if __name__ == '__main__':

    #create model
    fixed_params = {"p": .15, "c": .2, "issues": 3, "l_steps": 100}
    variable_params = {"n_agents":range(50,55,1)}
    batch_run = BatchRunner(
        bvmModel,
        variable_params,
        fixed_params,
        iterations=4,
        model_reporters = {'avg_assort':get_avg_assort, 'opinionClustering':returnNonUniform}
        )

    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    print(run_data)

    plotScatter(run_data, "n_agents", "avg_assort", "Number of Agents", "Assortativity")
