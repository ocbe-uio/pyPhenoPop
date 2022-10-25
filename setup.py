import io
import os
from setuptools import setup, find_packages

# Package meta-data.

NAME = 'mixpopid'
DESCRIPTION = 'Mixed leukemic tumor cell population identification '
URL = 'https://github.com/Apollinaria45/mixpopid'
EMAIL = 'apolinaria10@mail.ru'
AUTHOR = 'Polina Arsenteva'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '0.1.7'

REQUIRED = ['scipy', 'matplotlib','numpy']





here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        "Operating System :: OS Independent",
    ],
)
