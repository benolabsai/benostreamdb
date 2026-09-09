#!/usr/bin/env python
from setuptools import find_namespace_packages, setup

package_name = "dbt-hyperstreamdb"
package_version = "0.1.0"
description = """The HyperStreamDB adapter plugin for dbt"""

setup(
    name=package_name,
    version=package_version,
    description=description,
    long_description=description,
    author="Richard Albright",
    author_email="rla3rd@gmail.com",
    url="https://github.com/rla3rd/hyperstreamdb",
    packages=find_namespace_packages(include=["dbt", "dbt.*"]),
    include_package_data=True,
    install_requires=[
        "dbt-core>=1.8.0",
        "adbc-driver-flightsql>=1.12.0",
        "pyarrow>=15.0.0",
        "pandas",
    ],
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
