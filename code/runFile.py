from bvm_suite import *
from bvm_sweep import *
suite = bvmSuite({"p":.15, "c":.2, "issues":3, "l_steps":50, "n_agents":10}, 4)
suite.run()
print(suite.getData())

sweep = bvmSweep({"p":.15, "c":.2, "issues":3, "l_steps":50},{"n_agents":range(50,55,1)}, 4)

sweep.run()
print(sweep.getData())

