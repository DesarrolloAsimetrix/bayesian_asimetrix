from setuptools import setup, find_packages

def load_requirements(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return lines

def read_readme():
    with open('README.md', 'r') as f:
        return f.read()

def read_version():
    with open('VERSION', 'r') as f:
        return f.read()

setup(
    name="bayesian_asimetrix",
    version=read_version(),
    author="Asimetrix",
    author_email="info-analytics@asimetrix.co",
    description="Asimetrix Bayesian Inference Tools",
    long_description=read_readme(),
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt"),
)