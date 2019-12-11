from setuptools import setup
# For including non-python data, see:
# https://stackoverflow.com/a/1857436/4970632
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='climpy',
    url='https://github.com/lukelbd/climpy',
    author='Luke Davis',
    author_email='lukelbd@gmail.com',
    # Package stuff
    # Also include package data
    packages=['climpy'],
    # Needed for dependencies
    install_requires=['numpy', 'matplotlib', 'xarray', 'scipy'],
    # *Strongly* suggested for sharing
    version='0.0.2',
    # The license can be anything you like
    license='MIT',
    description='Toolset for working with climatological data.',
    long_description=open('README.rst').read(),
)
