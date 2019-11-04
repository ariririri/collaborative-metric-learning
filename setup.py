from setuptools import find_packages, setup

setup_requires = [
    "pytest-runner",
]
install_requires = [
    "numpy",
    "scipy",
    "torch",
    "tqdm",
    "scikit-learn",
]

tests_require = [
    'pytest',
    'pytest-cov',
    'pytest-flake8',
]

setup(
    name='cml',
    packages=find_packages(),
    version='0.2.0',
    description='machine learning modules',
    author='aririri',
    license='',
    install_requires=install_requires
)
