from operations import monomial, composite
import subprocess

#for key, value in {**monomial, **composite}.items():
subprocess.check_output(['python', 'train.py', '--operation=x'])
