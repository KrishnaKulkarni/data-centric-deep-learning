# Week 2 Project: Test driven deep learning

```
pip install -e .
```

This project will be using production deep learning tools to reliably evaluate deep learning models.

## Warmup

If you haven't yet, please do the [tutorial exercises](https://docs.metaflow.org/getting-started/tutorials) for MetaFlow. 

## Testing Accuracy Results

                    linear      mlp
Baseline        :   0.92        0.95
Integration     :   1.0         1.0
Regression      :   0.92        0.93 # It is odd that the linear model has this high a performance on the regression test examples, which were chosen to be the worst performing predictions for it
Directionality  :   0.69        0.78