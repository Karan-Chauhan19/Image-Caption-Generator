'''
author: Karan Chauuhan
github: @Karan-Chauhan19
organization: L.J University
'''

from setuptools import  setup, find_packages

setup(
    name='Image caption generator',  # project name
    version='1.0',  # version number
    description="Image caption generator using computer vision and natural langauge processing",  # description of the project
    packages = find_packages(),  # find all packages
    author='Karan-Chauhan' ,# author of the package
    author_email='kc879022@gmail.com', # email of the author
    url='https://github.com/Karan-Chauhan19/Image-Caption-Generator', # url of the project
    install_requires=['pandas','torch','tensorflow','keras'],
    # list of the dependencies required by the package
    classifiers=['Programming Language :: python :: 3.12.3']
)