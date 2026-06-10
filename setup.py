from setuptools import setup, find_packages

setup(
    name="guidance_rl",
    version="0.1.0",
    description="Learned guidance law for vision-based UAV interception (Phase 1)",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pyyaml>=5.4",
    ],
)
