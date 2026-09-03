import os

# The suite never touches a GPU: box etiquette made mechanical, not remembered.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
