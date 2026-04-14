import re
import os

with open("src/core/table/mod.rs", "r") as f:
    mod_lines = f.readlines()

def extract_between(start_str, end_str):
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(mod_lines):
        if start_str in line and start_idx == -1:
            # check if it's the right indentation (4 spaces)
            if line.startswith("    pub ") or line.startswith("    async fn ") or line.startswith("    fn "):
                start_idx = i
        if end_str in line and start_idx != -1:
            # find the end of that block
            pass # this is hard in python.

# Let's just use rustc --pretty !!
