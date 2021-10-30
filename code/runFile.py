from bvm_suite import *
from bvm_sweep import *
import pandas as pd
import numpy as np
import math


def cleanData(data):    
    dRange = np.arange(0.4, .95, 0.05)
    oRange = np.arange(0.05, .45, 0.05)
    bucketSum = 0
    sweepData = []
    labels = ['o', 'd', 'Buckets']
    for d in dRange:
        for o in oRange:
            o = round(o,2)
            d = round(d,2)
            if(o>=d):
                bucketSum=0
            else:
                data1 = data.loc[data['o']==o]
                data2 = data1.loc[data1['d']==d]
                bucketSum = data2['Buckets'].mean()
            #print("O:{} | D:{} | AvgBuckets: {}".format(o, d, bucketSum))
            sweepData.append([o, d, bucketSum])

    #print(sweepData)
    finalData = pd.DataFrame(sweepData, columns=labels)
    finalData.to_csv('CleanedCI2Data.csv')



#suite = bvmSuite({"p":.3, "o":.20, "d":.50,"issues":3, "l_steps":1000, "n_agents":50, 'CI2':True}, 25)

sweep = bvmSweep({"p":.3,"issues":3, "l_steps":1000, "n_agents":100, 'CI2':True},{"d":np.arange(0.4,.95,0.05), "o":np.arange(0.05,.45,0.05)}, 3)

#sweep.run()
#data = sweep.getData()

#data.to_csv('CI2SweepData.csv')
data = pd.read_csv('CI2SweepData.csv')
print(data)
cleanData(data)
sweep.plotBucketHeatmap("Cross Issue Influence", 'CleanedCI2Data.csv')




