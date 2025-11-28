import os
import subprocess
import sys

sfm_config_path = '/data/hd-map/OneX/B5-update-data/sfm-config.yaml'
mvs_config_path = '/data/hd-map/OneX/B5-update-data/mvs.yaml'

print(" sfm_config_path : ", sfm_config_path)

pSfM = subprocess.Popen(["/mnt/dev/mvs-SenseMap/SenseMap/bin/sfm/test_all_configure_outside", sfm_config_path])
pSfM.wait()

print(" mvs_config_path : ", mvs_config_path)

pPatchMatchThread = subprocess.Popen(["/mnt/dev/mvs-SenseMap/SenseMap/bin/mvs/test_patch_match", config_path])
pPatchMatchThread.wait()

pFusion = subprocess.Popen(["/mnt/dev/mvs-SenseMap/SenseMap/bin/mvs/test_fusion", config_path])
pFusion.wait()

pMeshing = subprocess.Popen(["/mnt/dev/mvs-SenseMap/SenseMap/bin/mvs/test_meshing", config_path])
pMeshing.wait()
