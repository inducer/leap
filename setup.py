#!/usr/bin/env python


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "leap/version.py"
    exec(compile(open(init_filename).read(), init_filename, "exec"), version_dict)

    setup(
        name="leap",
        version=version_dict["VERSION_TEXT"],
        description="Time integration by code generation",
        long_description=open("README.rst").read(),
        author="Andreas Kloeckner",
        author_email="inform@tiker.net",
        license="MIT",
        url="https://documen.tician.de/leap",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Other Audience",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Visualization",
            "Topic :: Software Development :: Libraries",
            "Topic :: Utilities",
        ],
        packages=find_packages(),
        python_requires="~=3.6",
        install_requires=[
            "numpy>=1.5",
            "pytools>=2014.1",
            "pymbolic>=2014.1",
            "pytest>=2.3",
            "dagrt>=2019.4",
            "mako",
        ],
    )


if __name__ == "__main__":
    main()
