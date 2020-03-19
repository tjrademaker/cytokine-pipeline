#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(10)  # deterministic random data
a = np.hstack((rng.normal(size=1000),
               rng.normal(loc=5, scale=2, size=1000)))
_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.gca().set_yscale('log')
plt.title("Histogram with 'auto' bins")
(0.5, 1.0, "Histogram with 'auto' bins")
plt.show()
