import os
import subprocess
import sys

img_path = "/data/dataset/Benchmark/facade/small_images"
workspace_path = "/data/dataset/Benchmark/facade"
vocab_path = "/data/3rdParty/colmap/vocab_tree_flickr100K_words1M.bin"

print("Using image dir     : ", img_path)
print("      workspace dir : ", workspace_path)
print("         vocab_path : ", vocab_path)

pClusterMapper = subprocess.Popen(["../bin/analysis", img_path, workspace_path, vocab_path])

pClusterMapper.wait()
