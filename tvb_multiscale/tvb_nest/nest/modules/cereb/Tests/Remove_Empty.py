import os
import sys

print("Removing Empty files")

if len(sys.argv) > 1:
    PathDir = sys.argv[1]

else:
    PathDir = "."

print("Looking into: " , PathDir)

for root,dirs,files in os.walk(PathDir):
    for name in files:

        filename=os.path.join(root,name)

        if os.stat(filename).st_size==0:
            print("Removing ", filename)
            os.remove(filename)
