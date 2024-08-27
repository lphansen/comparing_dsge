from setuptools import setup


setup(
    name='mfr_jupyterWidgets',

    version='0.1.1',

    description='',
    long_description='',

    author='Macro-Financial Modeling Team',
    author_email='jhuang12@uchicago.edu',

    license='Apache Software License',

    packages=['mfr.jupyterWidgets'],
    install_requires=['plotly', 'nbconvert>=5.3.1','jupyter>=1.0.0',
    'ipywidgets>=7.4.2'],
    zip_safe=False,
)
