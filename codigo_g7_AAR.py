# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:21:52 2021

@author: Sergio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import sklearn as sk

gd = pd.read_csv('data/gang.csv', index_col=0)
G = nx.DiGraph(gd.values)

plt.figure(figsize =(15, 15))
nx.draw_networkx(G, with_labels = True)

# DEGREE CENTRALITY
print("\nValores de Centralidad de grados:")
deg_centrality = nx.degree_centrality(G)
print(deg_centrality)

# CLOSENESS CENTRALITY
print("\nValores de Centralidad de cercanía:")
close_centrality = nx.closeness_centrality(G)
print(close_centrality)