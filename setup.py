# -*- coding: utf-8 -*-

from setuptools import setup
import subprocess
import os

PACKAGE = "vartests"

with open('README.md', encoding='utf-8') as f:
	long_description = f.read()

out = subprocess.Popen(['python', os.path.join('src', PACKAGE, 'version.py')], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout, _ = out.communicate()
version = stdout.decode("utf-8").strip()
print(version)

modules = \
[PACKAGE]
install_requires = \
['arch>=5.0.1,<6.0.0',
 'pandas>=1.3.4,<2.0.0',
 'pygosolnp>=2021.5.1,<2022.0.0',
 'tqdm>=4.62.3,<5.0.0']

setup_kwargs = {
    'name': PACKAGE,
    'version': version,
    'description': 'Statistic tests for Value at Risk (VaR) Models.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Rafael Rodrigues',
    'author_email': 'rafael.rafarod@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': f"https://github.com/rafa-rod/{PACKAGE}",
    'py_modules': modules,
    'include_package_data': True,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}

setup(**setup_kwargs)