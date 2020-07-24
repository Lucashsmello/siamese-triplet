from setuptools import setup

setup(name='siamese-triplet',
    version='0.8',
    description='Fork from adambielski/siamese-triplet',
    url='https://github.com/Lucashsmello/siamese-triplet',
    author='Lucas Mello',
    author_email='lucashsmello@gmail.com',
    license='public',
    packages=["siamese_triplet"],
    install_requires=[
        "torch",
        "numpy"
    ],
    zip_safe=False)
