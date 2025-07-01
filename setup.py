from setuptools import setup, find_packages

setup(
    name="awsllm_utils",
    version="0.1.0",
    packages=find_packages(include=["utils", "utils.*"]),
    install_requires=[],
    author="Seena",
    description="Utility module",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
