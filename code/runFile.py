from bvm_suite import *
from bvm_sweep import *
import pandas as pd
suite = bvmSuite({"p":.3, "o":.20, "d":.60,"issues":3, "l_steps":1000, "n_agents":50}, 100)
#suite.run()
#x = suite.getData()
#x.to_csv('suiteData.csv')
x = pd.read_csv('suiteData.csv')

print(x)

#suite.plotAvgClone_AntiCloneHist('suiteData.csv')
suite.plotAvgClone_AntiCloneScatter('suiteData.csv')
'''
sweep = bvmSweep({"p":.15, "o":.15, "d": .6,"issues":3, "l_steps":50},{"n_agents":range(50,55,1)}, 3)

sweep.run()
print(sweep.getData())
'''
