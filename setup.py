from setuptools import find_packages, setup

__version__ = "0.2.3rc1"
url = "https://github.com/ntt123/opax"

install_requires = ["pax3 >= 0.4.0rc3"]
setup_requires = []
tests_require = []

setup(
    name="opax",
    version=__version__,
    description="A stateful optimizer library for JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thông Nguyễn",
    url=url,
    keywords=["deep-learning", "jax", "pax", "optimizer", "adam", "sgd"],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(exclude=["examples", "tests"]),
    extras_require={"test": tests_require},
    python_requires=">=3.7",
)
