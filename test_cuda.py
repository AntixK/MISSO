import os
cmd = os.popen("nvidia-smi").read()
# s = os.system(cmd)
# print(s[0])
print(float(cmd[181:185]))
