from setuptools import setup, find_packages

setup(
    name='complex-FER',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'python-dotenv==1.0.0',
        'numpy==1.23.3',
        'pandas==1.4.3',
        'matplotlib==3.5.3',
        'scikit_learn==1.1.2',
        'tensorflow==2.9.2',
        'opencv-python==4.6.0.66',
        'deepface==0.0.79',
        'Pillow==9.5.0',
        'tabulate==0.8.10',
    ],
    entry_points={
        'console_scripts': [
            'complex-FER = src.complex_FER:main',
        ],
    },
)