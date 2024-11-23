from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Exploring einsum'
LONG_DESCRIPTION = 'Solely made to explore more about einstein summation operator.'

setup(
	name='eiengrad', 
	version=VERSION,
	author='Belati Jagad Bintang Syuhada',
	author_email='belatijagadbintangsyuhada@gmail.com',
	description=DESCRIPTION,
	long_description=LONG_DESCRIPTION,
	packages=find_packages(),
	install_requires=[
		'torch>=2.5',
		'tqdm>=4.6',
		'numpy>=2.1',
		'datasets>=3.1',
		'tokenizers>=0.2',
		'torchtext>=0.18',
	], 
)