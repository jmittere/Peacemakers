import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import time

from graspologic.cluster.autogmm import AutoGMMCluster
thing = AutoGMMCluster(max_iter=5)
x = np.random.uniform(0,10,(20,1))
start = time.time()
cluster = thing.fit(x)
print(f"That took {time.time() - start} milliseconds.")
print(f"AutoGMM thinks there are {cluster.n_components_} clusters (5 iter).")

predictions = thing.predict(np.arange(0,10,.1).reshape(100,1))

thing2 = AutoGMMCluster(max_iter=100)
start = time.time()
cluster2 = thing2.fit(x)
print(f"That took {time.time() - start} milliseconds.")
predictions2 = thing2.predict(np.arange(0,10,.1).reshape(100,1))
print(f"AutoGMM thinks there are {cluster2.n_components_} clusters. (100 iter)")
