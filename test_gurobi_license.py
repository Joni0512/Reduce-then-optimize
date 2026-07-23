"""Minimal Gurobi WLS license check - creates one trivial model, nothing else.
Run this on a cluster COMPUTE NODE (via sbatch), not the login node - login nodes often
have internet access that compute nodes lack, so a working test there would be misleading.
"""
import socket

import gurobipy as gp

print(f"Running on host: {socket.gethostname()}")
env = gp.Env()
model = gp.Model("license_check", env=env)
x = model.addVar(name="x")
model.setObjective(x, gp.GRB.MAXIMIZE)
model.addConstr(x <= 1)
model.optimize()
print(f"License OK - trivial model solved, x = {x.X}")
