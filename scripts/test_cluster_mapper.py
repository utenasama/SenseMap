import os
import subprocess
import sys

img_path = "/Users/sensetime/Project/dataset/SFM_Data/facade/images/small_images"
workspace_path = "/Users/sensetime/Project/dataset/SFM_Data/facade/images"

print("Using image dir     : ", img_path)
print("      workspace dir : ", workspace_path)

pTestClusterMapper = subprocess.Popen(["../bin/cluster_mapper_test", img_path, workspace_path])
pTestClusterMapper.wait()
