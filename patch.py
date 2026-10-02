with open("src/core/reader/scan.rs", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "let vector_indices: Vec<_> = self" in line:
        lines.insert(i, "println!(\"ALL INDEX FILES: {:?}\", self.config.index_files);\n")
        break

with open("src/core/reader/scan.rs", "w") as f:
    f.writelines(lines)
