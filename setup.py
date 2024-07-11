from setuptools import setup, find_packages

setup(
    name='SyncVision',
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'opencv-python',
        'scikit-learn',
        'scikit-image',
        'Pillow',
    ],
    author='Ashutosh Kumar',
    author_email='ak1825@rit.edu',
    description='A package for image registration using SIFT, ORB, and IntFeat.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ashu1069/MoonMetaSync',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
