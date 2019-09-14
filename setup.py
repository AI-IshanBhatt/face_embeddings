from setuptools import setup

with open("README.md") as fh:
    long_desc = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="face_embedding",
    description="find face embedding in an image in multiple ways",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Ishan Bhatt",
    packages=["face_embedding"],
    include_package_data=True,
    licence="Public",
    version="0.0.1",
    maintainer_email="ishan_bhatt@hotmail.com",
    test_suite="tests",
    python_requires='>=3.6',
    install_requires=requirements,
)
