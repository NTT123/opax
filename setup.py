from setuptools import find_packages, setup

__version__ = "0.1.3"
url = "https://github.com/ntt123/opax"

install_requires = ["pax"]
setup_requires = []
tests_require = []

setup(
    name="opax",
    version=__version__,
    description="A stateful optimizer library for Jax",
    author="Thông Nguyễn",
    url=url,
    keywords=["deep-learning", "jax", "pax", "optimizer", "adam", "sgd"],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(exclude=["examples", "tests"]),
    extras_require={"test": tests_require},
    python_requires=">=3.6",
)
