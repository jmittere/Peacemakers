from bvm_suite import *
from bvm_sweep import *

suite = bvmSuite({"p":.15, "c":.15, "d":.6,"issues":3, "l_steps":50, "n_agents":10}, 3)
suite.run()
print(suite.getData())

sweep = bvmSweep({"p":.15, "c":.15, "d": .6,"issues":3, "l_steps":50},{"n_agents":range(50,55,1)}, 3)

sweep.run()
print(sweep.getData())

