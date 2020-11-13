import os,sys,time
d = []
time.sleep(1)
f = open(sys.argv[1], "r")
for i, line in enumerate(f):
    d.append(line.strip().split())
    if i % 100000 == 0:
        print("loaded", i, "samples")
print("done!")
time.sleep(1000)
