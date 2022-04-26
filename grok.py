from operations import monomial, composite, other
import subprocess

for key, value in {**other, **monomial, **composite}.items():
    subprocess.check_output(['python', 'train.py', f'--operation={key}'])
