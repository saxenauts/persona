import pytest


pytestmark = pytest.mark.skip(
    reason="Legacy GraphOps v1 flow tests are incompatible with current v2 memory architecture"
)
