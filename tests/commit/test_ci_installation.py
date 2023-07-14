from unittest import TestCase

import ml4s
from ml4s.backend._backend import init_installed_backends


class TestCIInstallation(TestCase):

    def test_detect_tf_torch_jax(self):
        backends = init_installed_backends()
        names = [b.name for b in backends]
        self.assertIn('torch', names)
        self.assertIn('jax', names)
        self.assertIn('tensorflow', names)

    def test_verify(self):
        ml4s.verify()
