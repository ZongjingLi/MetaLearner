from setuptools import setup, find_packages

setup(
    name="rinarak",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="rinarak, the personal package for machine learning and more",

    # project main page

    # the package that are prerequisites
    packages=find_packages(),
    include_package_data = True,
    package_data={
        'rinarak': ['rinarak/*.grammar']
        },
)

"""
'':['moic',
        'moic/mklearn',
        'moic/learn/nn'],
        'moic': ['mklearn'],
        'bandwidth_reporter':['moic','moic/mklearn','moic/learn/nn']
               
"""