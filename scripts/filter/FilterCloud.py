import os
import subprocess
import sys

in_path = "fused.ply"
out_path = "fused_filtered.ply"
radiu_factor = "5"
thredhold = "5"

print("in  file  dir : ", in_path)
print("out file  dir : ", out_path)
print("radius factor : ", radiu_factor)
print("thredhold     : ", thredhold)

pClusterMapper = subprocess.Popen(["./build/FilterCloud", in_path, out_path, radiu_factor, thredhold])

pClusterMapper.wait()
