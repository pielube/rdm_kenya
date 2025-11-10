# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:00:20 2025

@author: ucbvplu
"""


import prim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.rand(1000, 3), columns=["x1", "x2", "x3"])
response = df["x1"]*df["x2"] + 0.2*df["x3"]

p = prim.Prim(df, response, threshold=0.5, threshold_type=">")

box = p.find_box()
box.show_tradeoff()

plt.show()