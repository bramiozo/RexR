import setuptools

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Bram van Es,...",
    author_email="bramiozo@gmail.com",
    name='omic_helpers',
    license="GNU GPLv3",
    description='OMIC helpers, for your multi faceted nightmares',
    version='0.0.8',
    long_description=README,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=['numpy', 'minepy', 'numba', 'scikit-learn', 'pandas', 'seaborn', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research',
    ],
)
