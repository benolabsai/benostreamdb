import h5py
import numpy as np

f = h5py.File('benchmarks/competitors/data/lastfm-64-dot.hdf5', 'r')
train = f['train'][:]
test = f['test'][:]
train_norms = np.linalg.norm(train, axis=1)
test_norms = np.linalg.norm(test, axis=1)

print("Train norms:", np.min(train_norms), np.max(train_norms), np.mean(train_norms), np.std(train_norms))
print("Test norms:", np.min(test_norms), np.max(test_norms), np.mean(test_norms), np.std(test_norms))
