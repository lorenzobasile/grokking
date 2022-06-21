from operations import monomial, composite, other
import subprocess

for key, value in {**other, **monomial, **composite}.items():
    if key=='rand':
        subprocess.check_output(['python', 'train.py', f'--operation={key}'])

#subprocess.check_output(['python', 'store_representations.py'])
