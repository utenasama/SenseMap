import os
import subprocess
import sys

workspace_path = "/Users/sensetime/Project/dataset/SFM_Data/shanxi_res/RunData0/community0001"
vocab_path = "/Users/sensetime/Project/COLMAP/vocab_tree_flickr100K_words1M.bin"

print("      workspace dir : ", workspace_path)
print("         vocab_path : ", vocab_path)

pTestMatching = subprocess.Popen(["../bin/test_matching", workspace_path, vocab_path])
pTestMatching.wait()
