#!/usr/bin/env python

"""
Bamboo
-----

bambo is a set of tools for pandas DataFrames.

"""
from __future__ import print_function

import os.path as op
import io
import sys
from setuptools import Command, setup, find_packages
from setuptools.command.test import test as TestCommand
from pip.req import parse_requirements

script_path = 'scripts'

install_reqs = parse_requirements('requirements.txt')

LICENSE = 'new BSD'


#long description
def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


setup_dict = dict(
    name='bamboo',
    version='0.1.0',
    description='Set of tools for pandas DataFrames',

    license='BSD 3-Clause',
    author='Borja Ayerdi',
    author_email='ayerdi.borja@gmail.com',
    maintainer='Borja Ayerdi',
    maintainer_email='ayerdi.borja@gmail.com',

    packages=find_packages(),

    install_requires=[str(ir.req) for ir in install_reqs],

    extra_files=['CHANGES.rst', 'LICENSE', 'README.rst'],

    long_description=read('README.rst', 'CHANGES.rst'),

    platforms='Linux/MacOSX',

    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved ::' + LICENSE,
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],

    extras_require={
        'testing': ['pytest'],
    }
)


#Python3 support keywords
if sys.version_info >= (3,):
    setup_dict['use_2to3'] = False
    setup_dict['convert_2to3_doctests'] = ['']
    setup_dict['use_2to3_fixers'] = ['']


class PyTest(TestCommand):

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup_dict.update(dict(tests_require=['pytest'],
                       cmdclass={'test': PyTest}))


if __name__ == '__main__':
    setup(**setup_dict)
