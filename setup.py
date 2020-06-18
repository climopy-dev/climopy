# For including non-python data, see:
# https://stackoverflow.com/a/1857436/4970632
from setuptools import setup
from os.path import exists

with open('requirements.txt') as f:
    install_req = [req.strip() for req in f.read().split('\n')]
install_req = [req for req in install_req if req and req[0] != '#']

classifiers = [
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.8',
]

if exists('README.rst'):  # when does this not exist?
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

setup(
    url='https://climopy.readthedocs.io',
    name='climopy',
    author='Luke Davis',
    author_email='lukelbd@gmail.com',
    maintainer='Luke Davis',
    maintainer_email='lukelbd@gmail.com',
    python_requires='>=3.8.0',
    project_urls={
        'Bug Tracker': 'https://github.com/lukelbd/climopy/issues',
        'Documentation': 'https://climopy.readthedocs.io',
        'Source Code': 'https://github.com/lukelbd/climopy'
    },
    packages=['climopy'],
    classifiers=classifiers,
    # normally uses MANIFEST.in but setuptools_scm auto-detects tracked files
    include_package_data=True,
    install_requires=install_req,
    license='MIT',
    description='A data analysis toolkit for climate scientists.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    use_scm_version={
        'version_scheme': 'post-release',
        'local_scheme': 'dirty-tag'
    },
    setup_requires=[
        'setuptools_scm',
        'setuptools>=30.3.0',
        'setuptools_scm_git_archive',
    ],
)
