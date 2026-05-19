"""
tests/test_network.py — pytest suite for nn.network (Module, Sequential, Network)

Run:  pytest tests/test_network.py -v
      pytest tests/test_network.py -v -k "xor"   # just the end-to-end test
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.value import Value
from nn.layer     import Layer
from nn.network   import Module, Network, Sequential


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers & shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════

EPS      = 1e-4
GRAD_TOL = 1e-3


def _scalar_loss(out: Value | list[Value]) -> Value:
    """Reduce any network output to a single scalar Value."""
    if isinstance(out, Value):
        return out
    total = out[0]
    for v in out[1:]:
        total = total + v
    return total


def numerical_grad(model: Sequential, x: list[float], idx: int) -> float:
    """Central-difference gradient estimate for parameters()[idx]."""
    p    = model.parameters()[idx]
    orig = p.data

    p.data = orig + EPS
    fp     = _scalar_loss(model(x)).data

    p.data = orig - EPS
    fm     = _scalar_loss(model(x)).data

    p.data = orig
    return (fp - fm) / (2 * EPS)


def analytical_grads(model: Sequential, x: list[float]) -> list[float]:
    model.zero_grad()
    loss = _scalar_loss(model(x))
    loss.backward()
    return [p.grad for p in model.parameters()]


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


@pytest.fixture
def tiny_net() -> Network:
    """2 → 4 → 1  tanh/linear, small weights to avoid saturation."""
    net = Network(
        [Layer(2, 4, 'tanh'), Layer(4, 1, 'linear')],
        name='tiny',
    )
    for p in net.parameters():
        p.data = 0.1
    return net


@pytest.fixture
def x2() -> list[float]:
    return [0.5, -0.3]


@pytest.fixture
def deep_net() -> Network:
    """3 → 8 → 8 → 8 → 1  for depth-related tests."""
    net = Network([
        Layer(3, 8, 'tanh'),
        Layer(8, 8, 'tanh'),
        Layer(8, 8, 'tanh'),
        Layer(8, 1, 'linear'),
    ], name='deep')
    for p in net.parameters():
        p.data = 0.05
    return net


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Module base class
# ═══════════════════════════════════════════════════════════════════════════════

class TestModule:

    def test_abstract_forward_raises(self):
        m = Module.__new__(Module)
        m.__init__()
        with pytest.raises(NotImplementedError):
            m.forward([1.0])

    def test_abstract_parameters_raises(self):
        m = Module.__new__(Module)
        m.__init__()
        with pytest.raises(NotImplementedError):
            m.parameters()

    def test_train_eval_toggle(self, tiny_net):
        assert tiny_net.training is True
        tiny_net.eval()
        assert tiny_net.training is False
        tiny_net.train()
        assert tiny_net.training is True

    def test_train_returns_self(self, tiny_net):
        assert tiny_net.train() is tiny_net

    def test_eval_returns_self(self, tiny_net):
        assert tiny_net.eval() is tiny_net

    def test_zero_grad_clears_all(self, tiny_net, x2):
        out = _scalar_loss(tiny_net(x2))
        out.backward()
        tiny_net.zero_grad()
        assert all(p.grad == 0.0 for p in tiny_net.parameters())

    def test_n_params_correct(self):
        # Layer(2,4): 4*(2+1)=12   Layer(4,1): 1*(4+1)=5   total=17
        net = Network([Layer(2, 4, 'tanh'), Layer(4, 1, 'linear')])
        assert net.n_params == 17

    def test_clip_grad_norm_scales_down(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        norm_before = tiny_net.grad_norm()
        max_norm    = norm_before / 2          # force a clip
        tiny_net.clip_grad_norm(max_norm)
        norm_after = tiny_net.grad_norm()
        assert norm_after <= max_norm + 1e-9

    def test_clip_grad_norm_no_op_when_small(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        max_norm = 1e6                          # effectively unlimited
        grads_before = [p.grad for p in tiny_net.parameters()]
        tiny_net.clip_grad_norm(max_norm)
        grads_after  = [p.grad for p in tiny_net.parameters()]
        for b, a in zip(grads_before, grads_after):
            assert abs(b - a) < 1e-12

    def test_clip_grad_norm_returns_unclipped_norm(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        expected_norm = tiny_net.grad_norm()
        returned_norm = tiny_net.clip_grad_norm(max_norm=1e6)
        assert abs(returned_norm - expected_norm) < 1e-9

    def test_clip_grad_value_clamps_each(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        clip = 0.001
        tiny_net.clip_grad_value(clip)
        for p in tiny_net.parameters():
            assert -clip <= p.grad <= clip

    def test_state_dict_length(self, tiny_net):
        sd = tiny_net.state_dict()
        assert len(sd) == tiny_net.n_params

    def test_state_dict_keys_are_strings(self, tiny_net):
        sd = tiny_net.state_dict()
        assert all(isinstance(k, str) for k in sd)

    def test_load_state_dict_restores_values(self, tiny_net):
        sd = tiny_net.state_dict()
        for p in tiny_net.parameters():
            p.data = 999.0            # corrupt weights
        tiny_net.load_state_dict(sd)
        for i, p in enumerate(tiny_net.parameters()):
            assert p.data == float(sd[str(i)])

    def test_load_state_dict_wrong_size_raises(self, tiny_net):
        with pytest.raises(ValueError, match="parameters"):
            tiny_net.load_state_dict({"0": 1.0})


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Sequential
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequential:

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError):
            Sequential([])

    def test_len(self):
        s = Sequential([Layer(2, 4), Layer(4, 1)])
        assert len(s) == 2

    def test_index_access(self):
        l0 = Layer(2, 4)
        l1 = Layer(4, 1)
        s  = Sequential([l0, l1])
        assert s[0] is l0
        assert s[1] is l1

    def test_name_access(self):
        l = Layer(2, 4)
        s = Sequential([l], names=['encoder'])
        assert s['encoder'] is l

    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            Sequential([Layer(2, 4), Layer(4, 1)], names=['x', 'x'])

    def test_mismatched_names_raises(self):
        with pytest.raises(ValueError):
            Sequential([Layer(2, 4), Layer(4, 1)], names=['only_one'])

    def test_auto_names(self):
        s = Sequential([Layer(2, 4), Layer(4, 1)])
        assert s.layer_names == ['layer_0', 'layer_1']

    def test_add_layer(self):
        s = Sequential([Layer(2, 4)])
        s.add(Layer(4, 1), name='output')
        assert len(s) == 2
        assert 'output' in s.layer_names

    def test_add_returns_self_for_chaining(self):
        s = Sequential([Layer(2, 4)])
        result = s.add(Layer(4, 1))
        assert result is s

    def test_add_duplicate_name_raises(self):
        s = Sequential([Layer(2, 4)], names=['hidden'])
        with pytest.raises(ValueError, match="already exists"):
            s.add(Layer(4, 1), name='hidden')

    def test_add_non_layer_raises(self):
        s = Sequential([Layer(2, 4)])
        with pytest.raises(TypeError):
            s.add("not_a_layer")

    def test_iter(self):
        layers = [Layer(2, 4), Layer(4, 1)]
        s      = Sequential(layers)
        assert list(s) == layers

    def test_forward_output_type_single(self, x2):
        s   = Sequential([Layer(2, 4, 'tanh'), Layer(4, 1, 'linear')])
        out = s(x2)
        assert isinstance(out, Value)

    def test_forward_output_type_multi(self, x2):
        s   = Sequential([Layer(2, 4, 'tanh'), Layer(4, 3, 'linear')])
        out = s(x2)
        assert isinstance(out, list) and len(out) == 3

    def test_forward_chaining_shape(self):
        s   = Sequential([Layer(3, 8, 'relu'), Layer(8, 4, 'tanh'), Layer(4, 2, 'linear')])
        out = s([0.1, 0.2, 0.3])
        assert isinstance(out, list) and len(out) == 2

    def test_parameters_flat(self):
        l0 = Layer(2, 4, bias=True)     # 4*(2+1) = 12
        l1 = Layer(4, 1, bias=True)     # 1*(4+1) = 5
        s  = Sequential([l0, l1])
        assert len(s.parameters()) == 17

    def test_parameters_are_originals(self):
        s = Sequential([Layer(2, 4)])
        s.parameters()[0].grad = 777.0
        assert s.layers[0].neurons[0].w[0].grad == 777.0
        s.parameters()[0].grad = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Network — construction & validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkConstruction:

    def test_basic_construction(self):
        net = Network([Layer(2, 4, 'tanh'), Layer(4, 1, 'linear')])
        assert len(net) == 2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            Network([Layer(2, 4, 'tanh'), Layer(99, 1, 'linear')])

    def test_custom_name(self):
        net = Network([Layer(2, 4), Layer(4, 1)], name='my-net')
        assert net.name == 'my-net'

    def test_default_name(self):
        net = Network([Layer(2, 4), Layer(4, 1)])
        assert net.name == 'Network'

    def test_custom_layer_names(self):
        net = Network(
            [Layer(2, 8, 'tanh'), Layer(8, 1, 'linear')],
            names=['encoder', 'output'],
        )
        assert net.layer_names == ['encoder', 'output']

    def test_param_count_2_4_1(self):
        # Layer(2,4,bias=True) = 4*3 = 12
        # Layer(4,1,bias=True) = 1*5 = 5
        net = Network([Layer(2, 4, bias=True), Layer(4, 1, bias=True)])
        assert net.n_params == 17

    def test_param_count_no_bias(self):
        # Layer(2,4,bias=False) = 4*2 = 8
        # Layer(4,1,bias=False) = 1*4 = 4
        net = Network([Layer(2, 4, bias=False), Layer(4, 1, bias=False)])
        assert net.n_params == 12


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Network — forward pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkForward:

    def test_single_output_is_value(self, tiny_net, x2):
        assert isinstance(tiny_net(x2), Value)

    def test_multi_output_is_list(self, x2):
        net = Network([Layer(2, 4, 'tanh'), Layer(4, 3, 'linear')])
        out = net(x2)
        assert isinstance(out, list) and len(out) == 3

    def test_output_is_finite(self, tiny_net, x2):
        assert math.isfinite(tiny_net(x2).data)

    def test_accepts_value_inputs(self, tiny_net, x2):
        inputs = [Value(v) for v in x2]
        out    = tiny_net(inputs)
        assert isinstance(out, Value)

    def test_gradient_flows_to_value_inputs(self, tiny_net, x2):
        v1, v2 = Value(x2[0]), Value(x2[1])
        out    = tiny_net([v1, v2])
        out.backward()
        assert v1.grad != 0.0, "grad did not reach v1"
        assert v2.grad != 0.0, "grad did not reach v2"

    def test_wrong_input_length_raises(self, tiny_net):
        with pytest.raises(ValueError):
            tiny_net([1.0, 2.0, 3.0])    # net expects 2 inputs

    def test_fresh_graph_each_call(self, tiny_net, x2):
        out1 = tiny_net(x2)
        out2 = tiny_net(x2)
        assert out1 is not out2

    def test_predict_returns_float(self, tiny_net, x2):
        result = tiny_net.predict(x2)
        assert isinstance(result, float)

    def test_predict_restores_training_mode(self, tiny_net, x2):
        tiny_net.train()
        tiny_net.predict(x2)
        assert tiny_net.training is True

    def test_predict_restores_eval_mode(self, tiny_net, x2):
        tiny_net.eval()
        tiny_net.predict(x2)
        assert tiny_net.training is False

    def test_predict_multi_output_returns_list(self, x2):
        net = Network([Layer(2, 4, 'tanh'), Layer(4, 3, 'linear')])
        out = net.predict(x2)
        assert isinstance(out, list) and len(out) == 3
        assert all(isinstance(v, float) for v in out)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Network — backward / gradient correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkBackward:

    def test_backward_populates_all_grads(self, tiny_net, x2):
        out = _scalar_loss(tiny_net(x2))
        out.backward()
        for i, p in enumerate(tiny_net.parameters()):
            assert p.grad != 0.0, f"param[{i}] has zero grad"

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu",
                                             "leaky_relu", "elu", "swish"])
    def test_finite_difference_2_layer(self, activation):
        """
        Central-difference check for every activation through a 2-layer net.
        Ground-truth test — catches any wrong backward formula that survived
        neuron-level testing.
        """
        random.seed(1)
        net = Network([
            Layer(3, 4, activation),
            Layer(4, 1, 'linear'),
        ])
        for p in net.parameters():
            p.data = 0.1
        x = [0.5, -0.3, 0.8]

        analytical = analytical_grads(net, x)
        for i in range(len(net.parameters())):
            num = numerical_grad(net, x, i)
            assert abs(analytical[i] - num) < GRAD_TOL, (
                f"{activation} param[{i}]: "
                f"analytical={analytical[i]:.6f}, numerical={num:.6f}"
            )

    def test_finite_difference_deep_net(self, deep_net):
        """Grad check through 4 layers — ensures chain rule accumulates correctly."""
        x = [0.3, -0.1, 0.7]
        analytical = analytical_grads(deep_net, x)
        for i in range(min(len(deep_net.parameters()), 24)):
            num = numerical_grad(deep_net, x, i)
            assert abs(analytical[i] - num) < GRAD_TOL, (
                f"deep_net param[{i}]: "
                f"analytical={analytical[i]:.6f}, numerical={num:.6f}"
            )

    def test_grad_accumulates_without_zero_grad(self, tiny_net, x2):
        analytical_grads(tiny_net, x2)
        g1 = [p.grad for p in tiny_net.parameters()]

        _scalar_loss(tiny_net(x2)).backward()   # no zero_grad
        g2 = [p.grad for p in tiny_net.parameters()]

        for a, b in zip(g1, g2):
            if a != 0.0:
                assert abs(b - 2 * a) < 1e-6

    def test_zero_grad_resets_network(self, tiny_net, x2):
        analytical_grads(tiny_net, x2)
        tiny_net.zero_grad()
        assert all(p.grad == 0.0 for p in tiny_net.parameters())

    def test_gradient_through_value_inputs(self):
        """Grad from the loss must flow all the way to raw input Value nodes."""
        random.seed(2)
        net = Network([Layer(2, 4, 'tanh'), Layer(4, 1, 'linear')])
        for p in net.parameters():
            p.data = 0.1

        v1, v2 = Value(0.5), Value(-0.3)
        out    = _scalar_loss(net([v1, v2]))
        out.backward()
        assert v1.grad != 0.0
        assert v2.grad != 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Network — diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkDiagnostics:

    def test_grad_norm_zero_before_backward(self, tiny_net):
        # gradients start at 0 → norm should be 0
        assert tiny_net.grad_norm() == 0.0

    def test_grad_norm_positive_after_backward(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        assert tiny_net.grad_norm() > 0.0

    def test_grad_norm_zero_after_zero_grad(self, tiny_net, x2):
        _scalar_loss(tiny_net(x2)).backward()
        tiny_net.zero_grad()
        assert tiny_net.grad_norm() == 0.0

    def test_weight_stats_keys(self, tiny_net):
        stats = tiny_net.weight_stats()
        assert set(stats.keys()) == set(tiny_net.layer_names)

    def test_weight_stats_fields(self, tiny_net):
        stats = tiny_net.weight_stats()
        for layer_stats in stats.values():
            assert {'mean', 'std', 'min', 'max', 'n'} == set(layer_stats.keys())

    def test_weight_stats_n_matches_param_count(self, tiny_net):
        stats = tiny_net.weight_stats()
        for layer_name, layer in tiny_net._layers.items():
            # weight_stats only counts weights, not biases
            n_weights = layer.n_inputs * layer.n_outputs
            assert stats[layer_name]['n'] == n_weights

    def test_summary_returns_string(self, tiny_net, capsys):
        result = tiny_net.summary()
        assert isinstance(result, str)
        assert 'tiny' in result

    def test_summary_contains_all_layer_names(self, capsys):
        net = Network(
            [Layer(2, 8, 'tanh'), Layer(8, 1, 'linear')],
            names=['hidden', 'output'],
        )
        s = net.summary()
        assert 'hidden' in s
        assert 'output' in s

    def test_repr_contains_name(self, tiny_net):
        assert 'tiny' in repr(tiny_net)

    def test_repr_contains_param_count(self, tiny_net):
        assert str(tiny_net.n_params) in repr(tiny_net)

    def test_repr_contains_mode(self, tiny_net):
        assert 'train' in repr(tiny_net)
        tiny_net.eval()
        assert 'eval' in repr(tiny_net)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Network — save / load
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkSaveLoad:

    def test_save_creates_file(self, tiny_net):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tiny_net.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_save_valid_json(self, tiny_net):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tiny_net.save(path)
            with open(path) as fp:
                data = json.load(fp)
            assert 'name' in data
            assert 'layers' in data
            assert 'state_dict' in data
        finally:
            os.unlink(path)

    def test_load_restores_weights(self, tiny_net):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            original_params = [p.data for p in tiny_net.parameters()]
            tiny_net.save(path)

            loaded = Network.load(path)
            loaded_params = [p.data for p in loaded.parameters()]
            assert original_params == loaded_params
        finally:
            os.unlink(path)

    def test_load_restores_architecture(self, tiny_net):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tiny_net.save(path)
            loaded = Network.load(path)
            assert loaded.n_params    == tiny_net.n_params
            assert len(loaded.layers) == len(tiny_net.layers)
            for l_orig, l_load in zip(tiny_net.layers, loaded.layers):
                assert l_orig.n_inputs  == l_load.n_inputs
                assert l_orig.n_outputs == l_load.n_outputs
                assert l_orig._activation_name == l_load._activation_name
        finally:
            os.unlink(path)

    def test_load_restores_name(self, tiny_net):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            tiny_net.save(path)
            loaded = Network.load(path)
            assert loaded.name == tiny_net.name
        finally:
            os.unlink(path)

    def test_load_restores_layer_names(self):
        net = Network(
            [Layer(2, 8, 'tanh'), Layer(8, 1, 'linear')],
            names=['encoder', 'decoder'],
        )
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            net.save(path)
            loaded = Network.load(path)
            assert loaded.layer_names == ['encoder', 'decoder']
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            Network.load('/nonexistent/path/model.json')

    def test_save_load_predictions_match(self, tiny_net, x2):
        """Loaded model must give identical predictions to the original."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            pred_orig = tiny_net.predict(x2)
            tiny_net.save(path)
            loaded    = Network.load(path)
            pred_load = loaded.predict(x2)
            assert abs(pred_orig - pred_load) < 1e-12
        finally:
            os.unlink(path)

    def test_save_load_roundtrip_bias_false(self):
        net = Network([
            Layer(2, 4, 'tanh',   bias=False),
            Layer(4, 1, 'linear', bias=False),
        ])
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            net.save(path)
            loaded = Network.load(path)
            assert loaded.n_params == net.n_params
            for l in loaded.layers:
                assert l.neurons[0].b is None
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  End-to-end training
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_single_sgd_step_reduces_loss(self, tiny_net, x2):
        """One gradient-descent step must make the loss smaller."""
        lr     = 0.01
        target = Value(1.0)

        pred         = _scalar_loss(tiny_net(x2))
        loss_before  = ((pred - target) ** 2).data

        tiny_net.zero_grad()
        pred2 = _scalar_loss(tiny_net(x2))
        loss  = (pred2 - target) ** 2
        loss.backward()
        for p in tiny_net.parameters():
            p.data -= lr * p.grad

        loss_after = ((_scalar_loss(tiny_net(x2)) - target) ** 2).data
        assert loss_after < loss_before, (
            f"Loss did not decrease: {loss_before:.6f} → {loss_after:.6f}"
        )

    def test_xor_convergence(self):
        """
        Full XOR training loop.
        Loss after 200 steps must be less than 10 % of the initial loss.
        XOR is the canonical sanity check for any backprop implementation.
        """
        random.seed(42)
        lr = 0.1

        model = Network([
            Layer(2,  8, 'tanh'),
            Layer(8,  8, 'tanh'),
            Layer(8,  1, 'linear'),
        ], name='xor')

        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [-1.0, 1.0, 1.0, -1.0]   # tanh targets in {-1, +1}

        def mse_loss():
            total = Value(0.0)
            for x, y in zip(xs, ys):
                pred  = _scalar_loss(model(x))
                total = total + (pred - Value(y)) ** 2
            return total

        loss_start = mse_loss().data

        for _ in range(200):
            loss = mse_loss()
            model.zero_grad()
            loss.backward()
            model.clip_grad_norm(max_norm=1.0)
            for p in model.parameters():
                p.data -= lr * p.grad

        loss_end = mse_loss().data

        assert loss_end < 0.1 * loss_start, (
            f"XOR did not converge: {loss_start:.4f} → {loss_end:.4f}"
        )

    def test_xor_predictions_correct_sign(self):
        """
        After sufficient training, the model must predict the correct
        *sign* for each XOR input.  (sign is all that matters for
        a ±1-target classification problem.)
        """
        random.seed(99)
        lr = 0.05

        model = Network([
            Layer(2, 16, 'tanh'),
            Layer(16, 8, 'tanh'),
            Layer(8,  1, 'linear'),
        ], name='xor-sign')

        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [-1.0, 1.0, 1.0, -1.0]

        for _ in range(300):
            total = Value(0.0)
            for x, y in zip(xs, ys):
                pred  = _scalar_loss(model(x))
                total = total + (pred - Value(y)) ** 2
            model.zero_grad()
            total.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

        for x, y in zip(xs, ys):
            pred = model.predict(x)
            assert (pred > 0) == (y > 0), (
                f"Wrong sign for x={x}: pred={pred:.4f}, target={y}"
            )

    def test_save_load_then_continue_training(self):
        """
        Save mid-training, load, continue — loss must keep decreasing.
        Validates that the loaded model is a faithful continuation point.
        """
        random.seed(11)
        lr = 0.1

        model = Network([Layer(2, 8, 'tanh'), Layer(8, 1, 'linear')])
        xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        ys = [-1.0, 1.0, 1.0, -1.0]

        def mse(m):
            t = Value(0.0)
            for x, y in zip(xs, ys):
                t = t + (_scalar_loss(m(x)) - Value(y)) ** 2
            return t

        # 50 steps before save
        for _ in range(50):
            loss = mse(model)
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data -= lr * p.grad

        loss_mid = mse(model).data

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            model.save(path)
            resumed = Network.load(path)

            # 50 more steps on the resumed model
            for _ in range(50):
                loss = mse(resumed)
                resumed.zero_grad()
                loss.backward()
                for p in resumed.parameters():
                    p.data -= lr * p.grad

            loss_final = mse(resumed).data
            assert loss_final < loss_mid, (
                f"Resumed training did not reduce loss: {loss_mid:.4f} → {loss_final:.4f}"
            )
        finally:
            os.unlink(path)