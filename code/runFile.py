from bvm_suite import *
from bvm_sweep import *
import pandas as pd
import numpy as np
suite = bvmSuite({"p":.3, "o":.20, "d":.60,"issues":3, "l_steps":1000, "n_agents":50}, 25)
suite.run()
x = suite.getData()
x.to_csv('suiteData.csv')
#x = pd.read_csv('suiteData.csv')

print(x)

#suite.plotAvgClone_AntiCloneHist('suiteData.csv')
suite.plotAvgClone_AntiCloneScatter('suiteData.csv')
'''
sweep = bvmSweep({"p":.3,"issues":3, "l_steps":1000, "n_agents":100},{"o":np.arange(0.05,0.45,0.05), "d":np.arange(0.30,0.85,0.05)}, 4)


#sweep.run()
#data = sweep.getData()
#data.to_csv('sweepData.csv')
data = pd.read_csv('sweepData.csv')
print(data)
sweep.plotBucketHeatmap('sweepData.csv')
'''
