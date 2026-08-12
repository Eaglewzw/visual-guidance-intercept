from setuptools import setup, find_packages

setup(
    name="guidance_rl",
    version="0.3.0",
    description="Learned-guidance and full-frame UAV interception",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "pyyaml>=5.4",
    ],
)
