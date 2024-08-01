# Week 3 Project: Active learning on fashion classifier

```
pip install -e .
```

This project will investigate various strategies to identify elements for relabeling.

# README
MODEL (DEFAULT).    : 30.2% {'acc': 0.3021000027656555, 'loss': 7.151782989501953}
MODEL (RANDOM)      : 41.6% {'acc': 0.41600000858306885, 'loss': 1.6307997703552246}
MODEL (UNCERTAINTY) : 37.6% {'acc': 0.37560001015663147, 'loss': 2.9340600967407227}
MODEL (MARGIN)      : 39.4% {'acc': 0.39409998059272766, 'loss': 1.798661708831787}
MODEL (ENTROPY)     : 41.1% {'acc': 0.4110000729560852, 'loss': 3.5284924507141113}
MODEL (AUGMENT w/ rotations): 59.7% {'acc': 0.5972000360488892, 'loss': 1.5563817024230957}
MODEL (AUGMENT w/ complex transforms): 59.2% {'acc': 0.5918999910354614, 'loss': 1.7650949954986572}