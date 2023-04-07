from setuptools import setup, find_packages

setup(
    name='complex-FER',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'absl == 0.0',
        'click == 8.1.3',
        'deepface == 0.0.79',
        'keras_tuner == 1.3.4',
        'matplotlib == 3.5.3',
        'numpy == 1.23.5',
        'pandas == 1.5.1',
        'Pillow == 9.5.0',
        'python-dotenv == 1.0.0',
        'scikit_learn == 1.2.2',
        'tabulate == 0.9.0',
        'tensorflow == 2.12.0',
    ],
    entry_points={
        'console_scripts': [
            'complex-FER = src:main',
        ],
    },
)