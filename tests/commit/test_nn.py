# import os; os.environ['CUDA_VISIBLE_DEVICES'] = ""
from unittest import TestCase

from unifyml import math
from unifyml.math import channel, batch
from unifyml import nn


LIBRARIES = ['jax', 'tensorflow', 'torch']


class TestNetworks(TestCase):

    def test_u_net_2d_network_sizes(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.u_net(2, 3, levels=3, filters=8, batch_norm=False, activation='ReLU', in_spatial=(64, 32))
            self.assertEqual(6587, nn.parameter_count(net), msg=lib)
            net_res = nn.u_net(2, 3, batch_norm=False, activation='SiLU', in_spatial=2, use_res_blocks=True)
            self.assertEqual(39059, nn.parameter_count(net_res), msg=lib)

    def test_u_net_3d_norm_network_sizes(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.u_net(2, 3, levels=3, filters=8, batch_norm=True, activation='Sigmoid', in_spatial=3)
            self.assertEqual(19707, nn.parameter_count(net), msg=lib)
            net_res = nn.u_net(2, 3, batch_norm=True, activation='SiLU', in_spatial=3, use_res_blocks=True)
            self.assertEqual(113939, nn.parameter_count(net_res), msg=lib)

    def test_u_net_1d_norm_network_sizes(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.u_net(2, 3, levels=2, filters=16, batch_norm=True, activation='tanh', in_spatial=1)
            self.assertEqual(5043, nn.parameter_count(net), msg=lib)
            net_res = nn.u_net(2, 3, batch_norm=True, activation='SiLU', in_spatial=1, use_res_blocks=True)
            self.assertEqual(14867, nn.parameter_count(net_res), msg=lib)

    def test_optimize_u_net(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.u_net(1, 1, levels=2)
            optimizer = nn.adam(net, 1e-3)
            net_res = nn.u_net(1, 1, levels=2, use_res_blocks=True, activation='SiLU')
            optimizer_2 = nn.sgd(net_res, 1e-3)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            def loss_function_res(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net_res, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))
                nn.update_weights(net_res, optimizer_2, loss_function_res, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))

    def test_optimize_u_net_jit(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.u_net(1, 1, levels=2)
            optimizer = nn.adagrad(net, 1e-3)
            net_res = nn.u_net(1, 1, levels=2, use_res_blocks=True, activation='SiLU')
            optimizer_2 = nn.rmsprop(net_res, 1e-3)

            @math.jit_compile
            def loss_function(x):
                print("Tracing loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            @math.jit_compile
            def loss_function_res(x):
                print("Tracing loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net_res, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))
                nn.update_weights(net_res, optimizer_2, loss_function_res, math.random_uniform(math.batch(batch=10), math.spatial(x=8, y=8)))

    def test_dense_net_network_params(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.dense_net(2, 3, layers=[10, 12], batch_norm=False, activation='ReLU')
            self.assertEqual(201, nn.parameter_count(net))
            params = nn.get_parameters(net)
            self.assertEqual(6, len(params))
            self.assertTrue(all(isinstance(p, math.Tensor) for p in params.values()))
            net = nn.dense_net(2, 3, layers=[10], batch_norm=True, activation='ReLU')
            self.assertEqual(83, nn.parameter_count(net), str(lib))

    def test_optimize_dense_net(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.dense_net(2, 3, layers=[10], batch_norm=True, activation='Sigmoid')
            optimizer = nn.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(batch(batch=10), channel(vector=2)))

    def test_optimize_invertible_conv_net(self):
        for lib in ['torch', 'tensorflow']:
            nn.use(lib)
            net = nn.invertible_net(3, 'conv_net', in_channels=2, batch_norm=True, activation='SiLU', layers=[])
            optimizer = nn.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_res_net(self):
        for lib in ['torch', 'tensorflow']:
            nn.use(lib)
            net = nn.invertible_net(3, 'res_net', in_channels=2, batch_norm=True, activation='SiLU', layers=[])
            optimizer = nn.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_u_net(self):
        for lib in ['torch', 'tensorflow']:
            nn.use(lib)
            net = nn.invertible_net(3, 'u_net', in_channels=2, batch_norm=True, activation='SiLU')
            optimizer = nn.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8)))

    def test_optimize_invertible_dense_net(self):
        for lib in ['torch', 'tensorflow']:
            nn.use(lib)
            net = nn.invertible_net(3, 'dense_net', in_channels=50, layers=[50])
            optimizer = nn.adam(net)

            def loss_function(x):
                print("Running loss_function")
                assert isinstance(x, math.Tensor)
                pred = math.native_call(net, x)
                return math.l2_loss(pred)

            for i in range(2):
                nn.update_weights(net, optimizer, loss_function, math.random_uniform(math.batch(batch=10), math.channel(c=50)))

    def test_invertible_net_network_sizes(self):
        for lib in ['torch', 'tensorflow']:
            nn.use(lib)
            net_u = nn.invertible_net(3, lambda: nn.u_net(2, 2, 4, 16, True, 'SiLU', 2))
            self.assertEqual(454296, nn.parameter_count(net_u))
            net_u = nn.invertible_net(3, 'u_net', in_channels=2)
            self.assertEqual(454296, nn.parameter_count(net_u))
            net_res = nn.invertible_net(3, 'res_net', in_channels=2, batch_norm=True, layers=[])
            self.assertEqual(1080, nn.parameter_count(net_res))
            net_conv = nn.invertible_net(3, 'conv_net', in_channels=2, batch_norm=True, layers=[])
            self.assertEqual(576, nn.parameter_count(net_conv))
            net_dense = nn.invertible_net(3, 'dense_net', in_channels=2, batch_norm=True, layers=[2])
            self.assertEqual(240, nn.parameter_count(net_dense))

    def test_conv_classifier(self):
        for lib in LIBRARIES:
            nn.use(lib)
            net = nn.conv_classifier(1, (2,), 1, blocks=[10], dense_layers=[], batch_norm=True, softmax=False, periodic=False)
            self.assertEqual(401, nn.parameter_count(net))

    # def test_fno(self):
    #     for lib in ['tensorflow', 'torch']:
    #         nn.use(lib)
    #         net = nn.fno(2, 1, 10, [8, 8, 8], batch_norm=False, in_spatial=2)
    #         x = math.random_uniform(math.batch(batch=10), math.channel(c=2), math.spatial(x=8, y=8))
    #         y = math.native_call(net, x)
    #         self.assertEqual(set(y.shape), set(math.batch(batch=10) & math.spatial(x=8, y=8)))
