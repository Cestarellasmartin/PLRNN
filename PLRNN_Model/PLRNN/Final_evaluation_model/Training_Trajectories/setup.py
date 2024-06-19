from setuptools import setup, find_packages

setup(
    name='BPTT_TF',
    version='0.1.0',
    description='Training (PL)RNNs for dynamical systems reconstruction using BPTT & Teacher Forcing.',
    author='Florian Hess, Max Ingo Thurm',
    author_email='Florian.Hess@zi-mannheim.de, Maxingo.thurm@zi-mannheim.de',
    url = "https://gitlab.zi.local/TheoNeuro/nonstationary-trial-data-autoencoder/-/tree/MIT_",
    packages=find_packages()
)