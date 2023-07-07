# Instructions:
1. Install Welder: https://sysdnn.visualstudio.com/_git/MemFusion
2. Test: python3 example.py



# Pattern: Cast + Gelu + Cast + Dropout + Matmul + BiasAdd

https://github.com/microsoft/torchscale/blob/main/torchscale/component/feedforward_network.py#L133-L137

## Forward:

```
[2048, 4096] = f([2048,16384], [4096, 16384], [4096], [2048,16384])
y = f(x, w, b, mask):
    x = x.cast(float32)
    x = x * 0.5 * (1 + erf(x * M_SQRT1_2)) # GELU
    x = x.cast(float16)
    x = x * mask /(1-p) #dropout(x, mask) * s
    y = x * w
    y = y + b
 
```

```
- einstein_v2(\"
m0[N0, N1] = input0[N0, N1].cast(`float32`); 
m1[N0, N1] = m0[N0, N1] * const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (m0[N0, N1] * const(0.70710678118654752440).cast(`float32`)).call(`erf`)); 
m2[N0, N1] = m1[N0, N1].cast(`float16`); 
m3[N0, N1] = m2[N0, N1] * input3[N0, N1] / const(0.5).cast(`float16`); 
m4[N0, N2] +=! m3[N0, N1] * input1[N2, N1];
output[N0, N2] = m4[N0, N2] + input2[N0];\", 
input_dict={ 
    \"input0\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]},  
    \"input1\" : { \"dtype\" : \"float16\", \"shape\" : [4096, 16384]},  
    \"input2\" : { \"dtype\" : \"float16\", \"shape\" : [4096]},  
    \"input3\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]}})
## @: tensorCoreConfig=(0, 1)
```


## Backward:

### db

```
[4096] = reduce_sum(2048, 4096)
db = f0(dy)
db = reduce_sum(dy, 0)
```

### dw

```
 [4096, 16384] = ([2048,16384], [2048, 4096], [2048,16384])
dw = f1(x, dy, mask)
    x = x.cast(float32)
    x = x * 0.5 * (1 + erf(x * M_SQRT1_2)) # GELU
    x = x.cast(float16)
    x = x * mask /(1-p) #dropout(x, mask) * s
    dw = dy^T * x # [4096, 16384] = [2048, 4096] * [2048,16384]
```

```
- einstein_v2(\"
m0[N0, N1] = input0[N0, N1].cast(`float32`); 
m1[N0, N1] = m0[N0, N1] * const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (m0[N0, N1] * const(0.70710678118654752440).cast(`float32`)).call(`erf`)); 
m2[N0, N1] = m1[N0, N1].cast(`float16`); 
m3[N0, N1] = m2[N0, N1] * input2[N0, N1] / const(0.5).cast(`float16`); 
output0[N2, N1] +=! input1[N0, N2] * m3[N0, N1];\", 
input_dict={ 
    \"input0\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]} ,  
    \"input1\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 4096]},  
    \"input2\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]}})
## @: tensorCoreConfig=(0, 1)
```

### dx
```
dx = f2(x, w, mask, dy)
    dx = dy * w  # [2048,16384]  = [2048, 4096] * [4096, 16384]
    dx = dx * (1-p) * mask
    dx = dx.cast(float32)
    #dx = dx * (cdf + x * pdf) # cdf = 0.5 * (1 + erf(x * M_SQRT1_2))  pdf = exp(-0.5 * x * x ) * M_2_SQRTPI * M_SQRT1_2 * 0.5;
    dx = dx * (0.5 * (1 + erf(x * M_SQRT1_2)) + x * (exp(-0.5 * x * x ) * M_2_SQRTPI * M_SQRT1_2 * 0.5))
```

```
- einstein_v2(\"
m0[N0, N1] +=! input3[N0, N2] * input1[N2, N1];
m1[N0, N1] = m0[N0, N1] * input2[N0, N1] * const(0.5).cast(`float16`);
m2[N0, N1] = m1[N0, N1].cast(`float32`); 
m3[N0, N1] = const(0.5).cast(`float32`) * (const(1.0).cast(`float32`) + (input0[N0, N1] * const(0.70710678118654752440).cast(`float32`)).call(`erf`));
m4[N0, N1] = (const(-0.5).cast(`float32`) * input0[N0, N1] * input0[N0, N1]).call(`exp`) * const(0.3989422804014327).cast(`float32`);
output0[N0, N1] = m2[N0, N1] * (m3[N0, N1] + input0[N0, N1] * m4[N0, N1]);
\", 
input_dict={
    \"input0\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]},  
    \"input1\" : { \"dtype\" : \"float16\", \"shape\" : [4096, 16384]},  
    \"input2\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 16384]},
    \"input3\" : { \"dtype\" : \"float16\", \"shape\" : [2048, 4096]}})
## @: tensorCoreConfig=(0, 1)
```



## reference:

M_SQRT1_2 = 0.70710678118654752440
M_2_SQRTPI = 1.12837916709551257390

https://github.com/pytorch/pytorch/blob/c24b61bc20f76c238e742b765a9efe9ae20c7c03/aten/src/ATen/native/cuda/ActivationGeluKernel.cu