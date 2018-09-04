import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(name = 'Fractals',
                 version = '0.1.dev0',
                 author = 'Hengjian Jia',
                 author_email = 'henryjia18@gmail.com',
                 description = 'Open Sourced Modularised Pytorch Production Framework',
                 long_description = long_description,
                 long_description_content_type = 'text/markdown',
                 url = 'https://github.com/HenryJia/Fractals',
                 license = 'MIT License',
                 packages = setuptools.find_packages(),
                 classifiers=['Programming Language :: Python :: 3',
                              'License :: OSI Approved :: MIT License',
                              'Operating System :: OS Independent']
                 )
