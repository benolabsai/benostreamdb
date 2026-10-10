import h5py
import numpy as np
f = h5py.File('benchmarks/competitors/data/lastfm-64-dot.hdf5', 'r')
test = f['test'][0]
train = f['train'][:]
gt = f['neighbors'][0][:10]

dots = np.dot(train, test)
top10 = np.argsort(dots)[-10:][::-1]
print("GT indices:", gt)
print("Manual top10:", top10)
print("Manual top10 dots:", dots[top10])

# check the retrieved indices from previous output
ret = [224086, 189959, 2821, 228060, 189960, 91348, 277177, 198285, 224743, 114163]
print("Retrieved dots:", dots[ret])

