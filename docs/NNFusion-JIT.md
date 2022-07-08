# How to use NNFusion JIT

NNFusion JIT is a JIT compiler for PyTorch using NNFusion CLI. It allows the user to JIT a function or a `torch.nn.Module` object using a decorator or a function wrapper. All NNFusion optimization such as kernel tuning can be applied using this interface.

## nnfusion.jit

`nnfusion.jit(obj=None, *, tune=None, tuning_steps=None, config=None)`

Lazily trace an object and optimize using nnfusion compilation until the object is called. It can be used as a decorator or an explicit function wrapper. The inputs and outputs of the object should be `torch.Tensor` or a sequence of `torch.Tensor`.

### Parameters

+ obj (function, `torch.nn.Module` instance / method / class):
    +  Target object to be traced. When `obj` is an instance or a class, it is equivalent to tracing its `forward` function.
+ tune (Optional[bool]):
    + Whether to tune kernel. By default it follows `config`. If set, it overwrites `config`.
+ tuning_steps (Optional[int]):
    + Number of kernel tuning steps. By default it follows `config`. If set, it overwrites `config` and `tune`.
            
+ config (Optional[dict, nnfusion.Config]):
    + NNFusion compilation config. By default it will be set to default config `nnfusion.Config()`. Pass a `dict` to overwrite default config or directly pass an instance of `nnfusion.Config`.
    + For example, `@nnfusion.jit(tune=True, config={'kernel_tuning_steps': 42})`
    + For more flags information, please execute the command `nnfusion` in the terminal.


### Use Cases Demo


`nnfusion.jit` can be used as a function wrapper for standalone functions and `torch.nn.Module` instances/methods/classes.


It can also be used as a decorator for standalone functions and `torch.nn.Module` methods/classes. 


```python
# Case 1: decorator for a standalone function
@nnfusion.jit
def foo(t1, t2):
    return t1 + t2, t1 - t2


# Case 2: decorator for a class method
class Net(nn.Linear):
    @nnfusion.jit
    def foo(self, x):
        return super().forward(x)

    
# Case 3: decorator for a class
@nnfusion.jit
class Net(nn.Linear):
    def this_will_not_be_traced(self, x):
        return super().forward(x)
    def forward(self, x):
        return super().forward(x)
    

# Case 4: function for a standalone function
def foo(t1, t2):
    return t1 + t2, t1 - t2
jitted_foo = nnfusion.jit(foo)


# Case 5: function for a torch.nn.Module class method
class Net(nn.Linear):
    def foo(self, x):
        return super().forward(x)
model = Net().eval()
model.foo = nnfusion.jit(model.foo)


# Case 6: function for a torch.nn.Module class 
jitted_linear = nnfusion.jit(nn.Linear)
model = jitted_linear().eval()


# Case 7: function for a torch.nn.Module instance
class Net(nn.Linear):
    def forward(self, x):
        return super().forward(x)
model = Net().eval()
jitted_model = nnfusion.jit(model)
```

It is allowed to pass optional keyword arguments:

```python
@nnfusion.jit(tune=True) 
def foo(t1, t2):
    return t1 + t2

def bar(t):
    return t + t, t * t
jitted_bar = nnfusion.jit(bar, tuning_steps=2000)
```

### Compiled kernels caching strategies

The compiled kernels are saved in `nnf-kernels/`. If a "match" kernel is found before the compilation, it will be directly reused. Here, "match" means having the same object signature (`__module__` and `__qualname__`), computational graph (ONNX model binary), and NNFusion compilation config (`config`).

## nnfusion.Config

`nnfusion.Config(*args, antares_mode=True, blockfusion_level=0, extern_result_memory=True, function_codegen=True, ir_based_fusion=False, kernel_fusion_level=0, kernel_tuning_steps=1000, **kwargs)`

NNFusion compilation config. Can pass in any other NNFusion compiler flags (execute the command `nnfusion` in the terminal for more details) and unknown flags will be ignored. Use it as a `dict` with some default key-value pairs.

### Use Cases Demo

```python
config = nnfusion.Config(function_codegen=False,
                         new_flag=42)
config = nnfusion.Config({'function_codegen': False,
                          'new_flag': 42})
config = nnfusion.Config()
config['function_codegen'] = False
config['new_flag'] = 42
```
