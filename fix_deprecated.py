import os

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    new_content = content.replace('set_thread_gpu_context', 'set_thread_gpu_context')
    new_content = new_content.replace('get_thread_gpu_context', 'get_thread_gpu_context')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

for root, _, files in os.walk('/home/ralbright/projects/hyperstreamdb'):
    if '/target/' in root or '/.git/' in root or '/venv/' in root or '/.venv' in root:
        continue
    for file in files:
        if file.endswith('.rs') or file.endswith('.py') or file.endswith('.md'):
            replace_in_file(os.path.join(root, file))
