# Copyright 2014 Jérôme Kieffer et al.
# This is an open-access article distributed under the terms of the
# Creative Commons Attribution License, which permits unrestricted use,
# distribution, and reproduction in any medium, provided the original author
# and source are credited.
# http://creativecommons.org/licenses/by/3.0/
# Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
# high performance azimuthal integration on gpu, 2014. In Proceedings of the
# 7th European Conference on Python in Science (EuroSciPy 2014).

import torch


# Implementation of histograms in PyTorch with CUDA support from
# https://github.com/francois-rozet/torchist/blob/d30c5742516e468852db5e186cfdf1fcb1a8994e/torchist/__init__.py
def ravel_multi_index(coords, shape):
    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return (coords * coefs).sum(dim=-1)

def quantize(x, bins, low, upp):
    x = (x - low) / (upp - low)  # in [0.0, 1.0]
    x = (bins * x).long()  # in [0, bins]
    return x

def histogramdd(x, bins=10, low=None, upp=None, bounded=False, weights=None):
    # Preprocess
    D = x.size(-1)
    x = x.reshape(-1, D)

    bounded = bounded or (low is None and upp is None)

    if low is None:
        low = x.min(dim=0).values

    if upp is None:
        upp = x.max(dim=0).values

    bins = torch.as_tensor(bins, dtype=torch.long, device=x.device).squeeze()
    low = torch.as_tensor(low, dtype=x.dtype, device=x.device).squeeze()
    upp = torch.as_tensor(upp, dtype=x.dtype, device=x.device).squeeze()

    assert torch.all(upp > low), "The upper bound must be strictly larger than the lower bound"

    if weights is not None:
        weights = weights.flatten()

    # Filter out-of-bound values
    if not bounded:
        mask = ~out_of_bounds(x, low, upp)

        x = x[mask]

        if weights is not None:
            weights = weights[mask]

    # Indexing
    idx = quantize(x, bins, low, upp)

    idx = torch.clip(idx, min=None, max=bins - 1)  # last bin includes upper bound

    # Histogram
    shape = torch.Size(bins.expand(D).tolist())

    idx = ravel_multi_index(idx, shape)
    hist = idx.bincount(weights, minlength=shape.numel()).reshape(shape)

    return hist

def histogram(x, bins=10, low=None, upp=None, **kwargs):
    return histogramdd(x.unsqueeze(-1), bins, low, upp, **kwargs)

def azimint_hist(data, radius, npt):
    histu = histogram(radius, npt)[0]
    histw = histogram(radius, npt, weights=data)[0]
    return histw / histu

azimint_hist_jit = torch.compile(azimint_hist)
