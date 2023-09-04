from setuptools import setup, find_packages

setup(
    name="FlexGP",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # List your package's dependencies here
    ],
    classifiers=[
        # Add classifiers to indicate the status, audience, and compatible Python versions
    ],
)
