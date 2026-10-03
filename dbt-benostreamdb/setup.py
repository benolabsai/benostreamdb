#!/usr/bin/env python
"""Thin shim: all metadata lives in pyproject.toml.

Kept so `pip install -e .` and legacy tooling that expects a setup.py still
work. The version is read from
`dbt/adapters/benostreamdb/__version__.py` via `[tool.setuptools.dynamic]`.
"""
from setuptools import setup

setup()
