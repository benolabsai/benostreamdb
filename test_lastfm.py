import h5py
import numpy as np

f = h5py.File('benchmarks/competitors/data/lastfm-64-dot.hdf5', 'r')
train = f['train'][:]
test = f['test'][:]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train_norms = np.linalg.norm(train, axis=1)
print("Train norms min/max/mean:", train_norms.min(), train_norms.max(), train_norms.mean())

test_norms = np.linalg.norm(test, axis=1)
print("Test norms min/max/mean:", test_norms.min(), test_norms.max(), test_norms.mean())

print("Train [0] last dim:", train[0][-1])
print("Test [0] last dim:", test[0][-1])
