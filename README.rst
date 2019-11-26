leap: Descriptive Time Integration with Flexbile Multi-Rate Algorithms
======================================================================

.. image:: https://gitlab.tiker.net/inducer/leap/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/leap/commits/master
.. image:: https://dev.azure.com/ak-spam/inducer/_apis/build/status/inducer.leap?branchName=master
    :alt: Azure Build Status
    :target: https://dev.azure.com/ak-spam/inducer/_build/latest?definitionId=21&branchName=master
.. image:: https://badge.fury.io/py/leap.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/leap/


leap describes both implicit and explicit `time stepping methods
<https://en.wikipedia.org/wiki/Time_stepping>`_. Methods are
described using a small virtual machine whose statements can
then be used to either generate code or simply execute the time
integrator.

Execution and code generation is provided by
`dagrt <https://github.com/inducer/dagrt>`__. Further
supported operations include finding of stability regions, building
of companion matrices, and more.

leap is licensed under the MIT license.

leap requires Python 2.6 or newer.

Resources:

* `Documentation <https://documen.tician.de/leap>`_
* `source code via git <https://github.com/inducer/leap>`_
