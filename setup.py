# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': '.'}

packages = \
['colour',
 'colour.adaptation',
 'colour.adaptation.datasets',
 'colour.adaptation.tests',
 'colour.algebra',
 'colour.algebra.coordinates',
 'colour.algebra.coordinates.tests',
 'colour.algebra.tests',
 'colour.appearance',
 'colour.appearance.tests',
 'colour.biochemistry',
 'colour.biochemistry.tests',
 'colour.blindness',
 'colour.blindness.datasets',
 'colour.blindness.tests',
 'colour.characterisation',
 'colour.characterisation.datasets',
 'colour.characterisation.datasets.cameras',
 'colour.characterisation.datasets.cameras.dslr',
 'colour.characterisation.datasets.colour_checkers',
 'colour.characterisation.datasets.displays',
 'colour.characterisation.datasets.displays.crt',
 'colour.characterisation.datasets.displays.lcd',
 'colour.characterisation.tests',
 'colour.colorimetry',
 'colour.colorimetry.datasets',
 'colour.colorimetry.datasets.illuminants',
 'colour.colorimetry.datasets.light_sources',
 'colour.colorimetry.tests',
 'colour.constants',
 'colour.continuous',
 'colour.continuous.tests',
 'colour.contrast',
 'colour.contrast.tests',
 'colour.corresponding',
 'colour.corresponding.datasets',
 'colour.corresponding.tests',
 'colour.difference',
 'colour.difference.tests',
 'colour.graph',
 'colour.graph.tests',
 'colour.io',
 'colour.io.luts',
 'colour.io.luts.tests',
 'colour.io.tests',
 'colour.models',
 'colour.models.datasets',
 'colour.models.rgb',
 'colour.models.rgb.datasets',
 'colour.models.rgb.tests',
 'colour.models.rgb.transfer_functions',
 'colour.models.rgb.transfer_functions.tests',
 'colour.models.tests',
 'colour.notation',
 'colour.notation.datasets',
 'colour.notation.datasets.munsell',
 'colour.notation.tests',
 'colour.phenomena',
 'colour.phenomena.tests',
 'colour.plotting',
 'colour.plotting.datasets',
 'colour.plotting.tests',
 'colour.quality',
 'colour.quality.datasets',
 'colour.quality.tests',
 'colour.recovery',
 'colour.recovery.datasets',
 'colour.recovery.tests',
 'colour.temperature',
 'colour.temperature.tests',
 'colour.utilities',
 'colour.utilities.tests',
 'colour.volume',
 'colour.volume.datasets',
 'colour.volume.tests']

package_data = \
{'': ['*'],
 'colour': ['examples/*',
            'examples/adaptation/*',
            'examples/algebra/*',
            'examples/appearance/*',
            'examples/blindness/*',
            'examples/characterisation/*',
            'examples/colorimetry/*',
            'examples/contrast/*',
            'examples/corresponding/*',
            'examples/difference/*',
            'examples/graph/*',
            'examples/io/*',
            'examples/io/resources/*',
            'examples/models/*',
            'examples/notation/*',
            'examples/phenomena/*',
            'examples/plotting/*',
            'examples/plotting/resources/*',
            'examples/quality/*',
            'examples/recovery/*',
            'examples/temperature/*',
            'examples/volume/*'],
 'colour.appearance.tests': ['fixtures/*'],
 'colour.io.luts.tests': ['resources/cinespace/*',
                          'resources/iridas_cube/*',
                          'resources/resolve_cube/*',
                          'resources/sony_spi1d/*',
                          'resources/sony_spi3d/*'],
 'colour.io.tests': ['resources/*']}

install_requires = \
['imageio', 'scipy', 'six']

extras_require = \
{'development': ['biblib-simple',
                 'coverage',
                 'coveralls',
                 'flake8',
                 'invoke',
                 'jupyter',
                 'mock',
                 'nose',
                 'pre-commit',
                 'pytest',
                 'restructuredtext-lint',
                 'sphinx',
                 'sphinx_rtd_theme',
                 'sphinxcontrib-bibtex',
                 'toml',
                 'twine',
                 'yapf==0.23'],
 'graphviz': ['pygraphviz'],
 'optional': ['networkx', 'pandas'],
 'plotting': ['backports.functools_lru_cache', 'matplotlib'],
 'read-the-docs': ['mock',
                   'networkx',
                   'numpy',
                   'pygraphviz',
                   'sphinxcontrib-bibtex']}

setup(
    name='colour-science',
    version='0.3.14',
    description='Colour Science for Python',
    long_description=open('README.rst').read(),
    author='Colour Developers',
    author_email='colour-developers@colour-science.org',
    maintainer='Colour Developers',
    maintainer_email='colour-developers@colour-science.org',
    url='https://www.colour-science.org/',
    package_dir=package_dir,
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
)
