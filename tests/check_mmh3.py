import mmh3
import sys

val = "iceberg"
h = mmh3.hash(val, seed=0, signed=False)
print(f"Hash of '{val}': {h}")

if h == 1216061395:
    print("MATCHES SPEC")
    sys.exit(0)
else:
    print(f"MISMATCH! Got {h}")
    # Also print signed
    h_signed = mmh3.hash(val, seed=0, signed=True)
    print(f"Signed: {h_signed}")
    sys.exit(1)
