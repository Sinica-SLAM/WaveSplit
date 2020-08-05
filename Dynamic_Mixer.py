import numpy as np
import os

dirs = os.listdir("wsj0/si_tr_s")

#for files in dirs:
#    print(files)

d = 'wsj0/si_tr_s'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
print(subdirs)