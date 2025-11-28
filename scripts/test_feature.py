import os
import subprocess
import sys

img_path = "/Users/sensetime/Project/dataset/SFM_Data/pipes/images/dslr_images"
workspace_path = "/Users/sensetime/Project/dataset/SFM_Data/pipes/images"

print("Using image dir     : ", img_path)
print("      workspace dir : ", workspace_path)

pTestFeature = subprocess.Popen(["../bin/test_feature", img_path, workspace_path])
pTestFeature.wait()