import re
import subprocess

out = subprocess.run("pstree -ap | grep python", shell=True, capture_output=True)
for line in out.stdout.decode().split("\n"):
    m = re.search("python3,(\d+) -cfrom multiprocessing.spawn import spawn_main;", line)
    if m:
        pid = m.group(1)
        print(f"[0] kill {pid}")
        subprocess.run(f"kill {pid}", shell=True)
    m = re.search("python3,(\d+) -c from multiprocessing.spawn import spawn_main;", line)
    if m:
        pid = m.group(1)
        print(f"[1] kill {pid}")
        subprocess.run(f"kill {pid}", shell=True)
    # m = re.search("python3,(\d)+ -c from multiprocessing.resource_tracker import main;", line)
    # if m:
    #     pid = m.group(1)
    #     print(f"[2] kill {pid}")
    #     subprocess.run(f"kill {pid}", shell=True)