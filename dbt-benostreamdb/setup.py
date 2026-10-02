#!/usr/bin/env python
import os
import re

from setuptools import find_namespace_packages, setup

package_name = "dbt-benostreamdb"


def _core_version() -> str:
    """Derive the version from the core engine's Cargo.toml (single source)."""
    cargo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Cargo.toml")
    try:
        with open(cargo, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'^version = "([^"]+)"', line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return "0.0.0"


package_version = _core_version()
description = """The BenoStreamDB adapter plugin for dbt"""

setup(
    name=package_name,
    version=package_version,
    description=description,
    long_description=description,
    author="Richard Albright",
    author_email="rla3rd@gmail.com",
    url="https://github.com/benolabsai/benostreamdb",
    packages=find_namespace_packages(include=["dbt", "dbt.*"]),
    include_package_data=True,
    install_requires=[
        "dbt-core>=1.8.0",
        "benostreamdb>=0.11.1",
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
