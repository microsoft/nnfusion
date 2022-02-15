import nnfusion
import pytest
import torch



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


def test_single_input_multi_identical_output():
    def func(t):
        return t, t
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)


def test_module_no_grad():
    model = torch.nn.Linear(8, 8).cuda().eval()
    for param in model.parameters():
        param.requires_grad = False
    t = torch.randn(1, 8, device="cuda")
    compare_torch_and_nrt(model, t)


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


def test_keep_signature_but_change_compute_graph():
    def func(t):
        return t + t
    t = torch.randn(8, device="cuda")
    compare_torch_and_nrt(func, t)

    def func(t):
        return t * t
    compare_torch_and_nrt(func, t)
