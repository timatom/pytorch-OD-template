from setuptools import setup, find_packages  # or find_namespace_packages

setup(
    # ...
    name='pytorch-od-template',
    version='0.1',
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['pytorch-od-template.data', 'pytorch-od-template.models'],  # ['*'] by default
        exclude=['pytorch-od-template.tests'],  # empty by default
    ),
    # ...
)