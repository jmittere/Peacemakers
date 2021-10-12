from graspologic.cluster.autogmm import AutoGMMCluster
import numpy as np
bob = np.random.uniform(0,1,10)
print(bob)
clusters = AutoGMMCluster().fit(bob)
