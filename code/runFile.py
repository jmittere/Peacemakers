from bvm_suite import *
from bvm_sweep import *

suite = bvmSuite({"p":.3, "c":.15, "d":.55,"issues":3, "l_steps":1000, "n_agents":100}, 75)

#suite.run()
#x = suite.getData()
#x.to_csv('suiteData.csv')
#print(x)
suite.plotAvgClone_AntiCloneHist('suiteData.csv')

'''
sweep = bvmSweep({"p":.15, "c":.15, "d": .6,"issues":3, "l_steps":50},{"n_agents":range(50,55,1)}, 3)

sweep.run()
print(sweep.getData())
'''
