import os

# Every pytest run (tests and claims alike) stays off the GPU: box etiquette made mechanical.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
