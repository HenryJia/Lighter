import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='Lighter',
    version='0.1.dev0',
    author='Hengjian Jia',
    author_email='henryjia18@gmail.com',
    description='A high level library of deep learning tools built on top of PyTorch inspired by Ignite',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HenryJia/Lighter',
    license='MIT License',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent'])
