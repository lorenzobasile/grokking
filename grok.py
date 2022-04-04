from operations import ops
import subprocess

for key, value in ops.items():
    subprocess.check_output(['python', 'train.py', f'--operation={key}'])
