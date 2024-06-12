from setuptools import setup


setup(
    name='mfr_sdm',

    version='0.1.1',

    description='',
    long_description='',

    author='Macro-Financial Modeling Team',
    author_email='jhuang12@uchicago.edu',

    license='Apache Software License',

    packages=['mfr.sdm'],
    zip_safe=False,
    install_requires=['numba>=0.42.0', 'pyMKL'],
)
