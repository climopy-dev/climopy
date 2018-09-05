from setuptools import setup
# For including non-python data, see:
# https://stackoverflow.com/a/1857436/4970632
setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name         = 'AtmosData',
    url          = 'https://github.com/lukelbd/atmosdata',
    author       = 'Luke Davis',
    author_email = 'lukelbd@gmail.com',
    # Package stuff
    # Also include package data
    packages     = ['atmosdata'],
    # Needed for dependencies
    install_requires = ['numpy', 'matplotlib', 'xarray', 'scipy'],
    # *Strongly* suggested for sharing
    version = '0.1',
    # The license can be anything you like
    license          = 'LICENSE.txt',
    description      = 'Tricked out matplotlib wrapper for making clear, compact, publication-quality graphics quickly and easily.',
    long_description = open('README.md').read(),
)
