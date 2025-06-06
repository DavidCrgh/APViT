from setuptools import setup, find_packages

setup(
    name='mmcls',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'mmcv',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)