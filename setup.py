from setuptools import find_packages, setup


def _get_version():
    with open("opax/__init__.py", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("__version__"):
                g = {}
                exec(line, g)  # pylint: disable=exec-used
                return g["__version__"]
        raise ValueError("`__version__` not defined in `opax/__init__.py`")


__version__ = _get_version()

url = "https://github.com/ntt123/opax"

install_requires = ["pax3>=0.4.3"]
setup_requires = []
tests_require = []

setup(
    name="opax",
    version=__version__,
    description="A stateful optimizer library for JAX",
    long_description=open("README.md", encoding="utf-8").read(),
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
