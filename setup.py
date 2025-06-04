from setuptools import setup, find_packages

setup(
    name='mmcls',
    version='0.1.0',
    packages=['mmcls'],
    package_dir={'mmcls': '.'},
    install_requires=[
        'mmcv',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)