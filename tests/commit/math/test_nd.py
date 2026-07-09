from itertools import product
from unittest import TestCase
from phiml import math
from phiml.math import wrap, extrapolation, PI, tensor, batch, spatial, instance, channel, NAN, vec
from phiml.math.nd import smooth

import numpy as np
import os


REF_DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reference_data')


class TestMathNDNumpy(TestCase):

    def test_smooth_1d_kernels(self):
        impulse = tensor([0., 0., 1., 0., 0.], spatial('x'))
        math.assert_close([0.05448869, 0.24420136, 0.40261996, 0.24420136, 0.05448869], smooth(impulse, 1.0, 'gaussian', extrapolation.ZERO))
        math.assert_close([0.06745081, 0.1833503, 0.4983978, 0.1833503, 0.06745081], smooth(impulse, 1.0, 'laplace', extrapolation.ZERO))
        math.assert_close([0., 0.25, 0.5, 0.25, 0.], smooth(impulse, 1.0, 'linear', extrapolation.ZERO))

    def test_smooth_zero_radius_identity(self):
        values = tensor([0., 1., 2., 3., 4.], spatial('x'))
        math.assert_close(values, smooth(values, 0.0, 'linear', extrapolation.ZERO))
        math.assert_close(values, smooth(values, 0.0, 'gaussian', extrapolation.ZERO))

    def test_smooth_anisotropic_and_selected_dims(self):
        impulse = np.zeros((9, 9), np.float32)
        impulse[4, 4] = 1.
        impulse = tensor(impulse, spatial('x,y'))
        anisotropic = smooth(impulse, vec(x=1.0, y=2.0), 'gaussian', extrapolation.ZERO)
        self.assertEqual(anisotropic.shape, impulse.shape)
        self.assertAlmostEqual(anisotropic.numpy(('x', 'y')).sum(), 1.0, places=6)
        x_only = smooth(impulse, 1.0, 'gaussian', extrapolation.ZERO, dims='x')
        math.assert_close([1.3383062e-04, 4.4318615e-03, 5.3991124e-02, 2.4197143e-01, 3.9894345e-01, 2.4197143e-01, 5.3991124e-02, 4.4318615e-03, 1.3383062e-04], x_only.y[4])
