# Transformer Block

GPT-2 Architecture has 12 transformer blocks.

Each block consists of the following:
- Attention
- MLP (multi-layer perceptron)
- LayerNorm x2 (LayerNorm1, LayerNorm2)

We have the following classes:

`Block`
    - `Attention`
    - `Mlp`
    - `LayerNorm` - 2 of them
    - `forward` - method that takes in an `x` and outputs result `h`


## Forward Pass Implementation

It is just this:

```python
x = x + attn(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

