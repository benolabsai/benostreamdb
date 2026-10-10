import h5py
import numpy as np

f = h5py.File('benchmarks/competitors/data/lastfm-64-dot.hdf5', 'r')
test = f['test'][0]
train = f['train'][:]
gt = [161850, 107119, 46258, 144310, 136336, 76757, 150177, 165938, 255226, 255141]
ret = [224086, 189959, 2821, 228060, 189960, 91348, 277177, 198285, 224743, 114163]

def fwht(a):
    h = 1
    tmp = a.copy()
    while h < len(tmp):
        for i in range(0, len(tmp), h * 2):
            for j in range(i, i + h):
                x = tmp[j]
                y = tmp[j + h]
                tmp[j] = x + y
                tmp[j + h] = x - y
        h *= 2
    return tmp

def pad_pow2(a):
    n = 1
    while n < len(a): n *= 2
    return np.pad(a, (0, n - len(a)))

# Quantizer parameters from train[0:10000]
train_sample = train[:10000]
rotated = np.array([fwht(pad_pow2(x)) for x in train_sample])
global_min = rotated.min()
global_max = rotated.max()
scale = 255.0 / (global_max - global_min)
offset = global_min

def encode(x, c=1.0):
    r = fwht(pad_pow2(x * c))
    q = np.round((r - offset) * scale)
    return np.clip(q, 0, 255).astype(np.uint8)

train_u8 = np.array([encode(x) for x in train[gt + ret]])

# NORMALIZE QUERY!
test_norm = test / np.linalg.norm(test)

# Calculate c for query
rot_test = fwht(pad_pow2(test_norm))
q_min = rot_test.min()
q_max = rot_test.max()
quantizer_min = offset
quantizer_max = offset + 255.0 / scale
c = float('inf')
if q_min < -1e-6 and quantizer_min <= 0.0:
    c = min(c, quantizer_min / q_min)
if q_max > 1e-6 and quantizer_max >= 0.0:
    c = min(c, quantizer_max / q_max)
c *= 0.99
print("c:", c)

test_u8 = encode(test_norm, c)

def l2u8(a, b):
    return np.sum((a.astype(np.int32) - b.astype(np.int32))**2)

print("GT u8 distances:")
for i, idx in enumerate(gt):
    print(f"GT[{idx}] dist: {l2u8(test_u8, train_u8[i])} (dot: {np.dot(test, train[idx])})")

print("\nRet u8 distances:")
for i, idx in enumerate(ret):
    print(f"Ret[{idx}] dist: {l2u8(test_u8, train_u8[10 + i])} (dot: {np.dot(test, train[idx])})")

