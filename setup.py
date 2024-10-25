
from setuptools import setup, find_packages

setup(
    name="ailibrary",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.65.0",
    ],
    author="AILibrary Team",
    author_email="team@ailibrary.org",
    description="An accessible AI framework for assistive technology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ailibrary",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Accessibility",
    ],
    python_requires=">=3.8",
)
