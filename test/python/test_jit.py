import pytest
import torch
from torch.nn import functional as F

import nnfusion


def assert_allclose(output1, output2, rtol=1e-5, atol=1e-8):

    if not isinstance(output1, (tuple, list)):
        assert not isinstance(output2, (tuple, list))
        assert output1.size() == output2.size()
        assert torch.allclose(output1, output2, rtol, atol), (
            output1, output2, output1.sub(output2).max())
        return

    assert isinstance(output2, (tuple, list))
    assert len(output1) == len(output2)

    for out1, out2 in zip(output1, output2):
        assert_allclose(out1, out2, rtol, atol)


def compare_torch_and_nrt(obj, *inputs, step=1, run=None):
    assert step >= 1

    if step == 1:
        result_torch = obj(*inputs)
        result_nrt = nnfusion.jit(obj)(*inputs)
    else:
        def repeat(obj, *inputs):
            for _ in range(step):
                outputs, inputs = run(obj, *inputs)
            return outputs

        assert run is not None
        result_torch = repeat(obj, *inputs)
        result_nrt = repeat(nnfusion.jit(obj), *inputs)

    assert_allclose(result_torch, result_nrt)


def linear(input, weight, bias):
    return F.linear(input, weight, bias)


def test_single_input_multi_output():
    def func(t):
        return t + t, t * t
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)


def test_multi_input_single_output():
    def func(t1, t2):
        return t1 + t2
    t = [torch.randn(8, device="cuda") for _ in range(2)]
    compare_torch_and_nrt(func, *t)


def test_multi_identical_input_single_output():
    def func(t1, t2):
        return t1 + t2
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t, t)


@pytest.mark.xfail(reason=(
    "Probably identical tensors are fused during optimization. "
    "May need a copy at backend"))
def test_single_input_multi_identical_output():
    def func(t):
        return t, t
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)


@pytest.mark.xfail(reason="Compilation Error")
def test_single_input_multi_identical_output_advanced():
    def func(t):
        t2 = t + t
        return t2, t2
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)


def test_module_no_grad():
    model = torch.nn.Linear(8, 8).cuda().eval()
    t = torch.randn(1, 8, device="cuda")
    compare_torch_and_nrt(model, t)


def test_jit_class_method_using_decorator():
    class Foo(torch.nn.Linear):
        @nnfusion.jit
        def foo(self, t):
            return t + t
        @nnfusion.jit
        def bar(self, t):
            return self.forward(t) + 1

    model = Foo(8, 8).cuda().eval()
    t = torch.randn(1, 8, device="cuda")
    assert_allclose(t + t, model.foo(t))
    assert_allclose(linear(t, model.weight, model.bias) + 1, model.bar(t))

    class Bar(torch.nn.Linear):
        @nnfusion.jit
        def forward(self, t):
            return super().forward(t)
    model = Bar(8, 8).cuda().eval()
    assert_allclose(linear(t, model.weight, model.bias), model(t))


def test_jit_class_method_using_function():
    class Foo(torch.nn.Linear):
        def foo(self, t):
            return self.forward(t) + 1

    t = torch.randn(1, 8, device="cuda")
    model = Foo(8, 8).cuda().eval()
    assert_allclose(linear(t, model.weight, model.bias) + 1, model.foo(t))


def test_jit_class_using_decorator():
    def func(t):
        return t + t

    @nnfusion.jit
    class Foo(torch.nn.Linear):
        @nnfusion.jit
        def foo(self, t):
            return func(t)

    model = Foo(8, 8).cuda().eval()
    t = torch.randn(1, 8, device="cuda")
    assert_allclose(linear(t, model.weight, model.bias), model(t))
    assert_allclose(func(t), model.foo(t))


def test_jit_class_using_function():
    LinearJIT = nnfusion.jit(torch.nn.Linear)

    model = LinearJIT(8, 8).cuda().eval()
    t = torch.randn(1, 8, device="cuda")
    assert_allclose(linear(t, model.weight, model.bias), model(t))


@pytest.mark.xfail(reason=(
    "nnfusion codegen and compile success exit with 0 "
    "but para_info.json is null"))
def test_nested_jit():
    @nnfusion.jit
    def func1(t): return t + t

    @nnfusion.jit
    def func2(t): return func1(t) + 1

    t = torch.randn(1, 8, device="cuda")
    assert_allclose(t + t, func1(t))
    assert_allclose(t + t + 1, func2(t))


@pytest.mark.parametrize("step", [1, 5])
def test_repeat(step):
    def run(func, *inputs):
        outputs = func(*inputs)
        next_inputs = outputs
        return outputs, next_inputs

    def func(t1, t2):
        return t1 + t2, t1 - t2

    t = [torch.randn(8, device="cuda") for _ in range(2)]
    compare_torch_and_nrt(func, *t, step=step, run=run)


@pytest.mark.xfail(reason=(
    "Probably some bug when calling nnfusion to compile the same kernel "
    "in **a single process** (works well for not a single process). "
    "Not likely to happen in general use (but should work)."))
def test_keep_signature_but_change_compute_graph():
    def func(t):
        return t + t
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)

    # just to show that it can pass for compiling another function
    # TODO delete after fixing bug
    def func2(t):
        return t * 8
    compare_torch_and_nrt(func2, t)

    # same as the first one (identical jit-signature)
    # to ensure the compiled kernel will be regenerated if graph don't match
    def func(t):
        return t * t
    compare_torch_and_nrt(func, t)
