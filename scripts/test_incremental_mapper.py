import os
import subprocess
import sys

img_path = "/Users/sensetime/Project/dataset/SFM_Data/facade/images/small_images"
workspace_path = "/Users/sensetime/Project/dataset/SFM_Data/facade/images"

print("Using image dir     : ", img_path)
print("      workspace dir : ", workspace_path)

pTestIncrementMapper = subprocess.Popen(["../bin/incremental_mapper_test", img_path, workspace_path])
pTestIncrementMapper.wait()