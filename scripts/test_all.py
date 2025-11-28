import os
import subprocess
import sys

img_path = "/home/zhouliyang/Project/dataset/RunData0/images"
workspace_path = "/home/zhouliyang/Project/dataset/RunData0"
vocab_path = "/home/zhouliyang/Project/COLMAP/vocab_tree_flickr100K_words1M.bin"

print("Using image dir     : ", img_path)
print("      workspace dir : ", workspace_path)
print("         vocab_path : ", vocab_path)

pClusterMapper = subprocess.Popen(["../bin/test_all", img_path, workspace_path, vocab_path])

pClusterMapper.wait()
