from setuptools import setup, find_packages

setup(
    name="smaclite",
    version="0.0.1",
    description="SMAClite environment",
    author="Adam Michalski",
    url="https://github.com/micadam/smaclite",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["numpy", "gym>=0.12", "pygame"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)