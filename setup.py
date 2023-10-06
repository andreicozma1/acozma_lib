from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="acozma",
    version="0.1",
    author="Andrei Cozma",
    author_email="andreig992@gmail.com",
    # description="TODO",
    license="Apache",
    url="https://github.com/andreicozma1/acozma_lib/",
    packages=find_packages(include=["acozma"]),
    install_requires=required,
    python_requires=">=3.8.0",
)
