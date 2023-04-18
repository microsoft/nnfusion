### Recursion

A recursive function that allows the nodes inside its body graph "call" this node.

##### Attributes:

**`body`: graph (required)**

​	The graph run in each recursive call. It has N inputs and M outputs, matching the N inputs and M outputs of the defined recursive node. Its name may be used as an operator by its nodes, defining a recursive call to the graph.

##### Inputs (0 - inf)

**`args` (variadic, heterogeneous): V**

​	The inputs to the body graph

##### Outputs (0 - inf)

**`results` (variadic, heterogeneous): V**

​	the output of the body graph

##### Type constraints

**`V`: tensor(*)**

​	All Tensor types

##### Examples

`frac(x)`, which computes $x!$

```python
def frac(x):
	if x == 1:
		y = 1
    else:
		y = frac(x-1)
   	z = x * y
    return z
```

The equivalent graph (a demo, not tested):
```
graph frac_func_full (
  %x[INT64, scalar]
  %cond[BOOL, scalar]
) {
  %z = Recursion[body = <graph frac_node>](%x)
  return %y
}

graph frac_node (
  %x[INT64, scalar]
) {
  %c1 = Constant[value = <Scalar Tensor [1]>]()
  %cond = Equal(%x, %c1)
  %y = If[else_branch = <graph else_node>, then_branch = <graph then_node>](%cond)
  %z = Mul(%x, %y)
  return %z
}

graph else_node {
  %c1_int = Constant[value_int = 1]()
  %x_1 = Sub(%x, %c1_int)
  %y_1 = frac_node(%x)
  return %y_1
}

graph then_node {
  %y_0 = Constant[value_int = 1]()
  return %y_0
}
```
