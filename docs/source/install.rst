Installation
===================================

GitHub
------------

Currently, **Morphelia** is only available from GitHub.
Clone the repository.

.. code-block:: console

    cd /your/path
    git clone https://github.com/marx-alex/Morphelia

Create a new Conda environment.
Use morphelia_cpu.yml if GPU is not available on your machine.

.. code-block:: console

    (base) conda env create -f morphelia.yml -n morphelia
    (base) conda activate morphelia

Install **Morphelia**.

.. code-block:: console

    (morphelia) python -m pip install .
