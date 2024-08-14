# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:30:09 2024

@author: dienu
"""
import numpy as np

list1 = [1, 2, 3, 4, 5, 6]
list2 = ['a', 'b', 'c', 'd', 'e', 'f']
list3 = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

# Create a grid of all possible combinations
grid = np.array(np.meshgrid(list1, list2, list3)).T.reshape(-1, 3)
''' -1 means that the size of this dimension is calculated from the length of the
array and the specified number of columns (3 in this case).'''

print(f"This is waht the grid looks like and its length:\n{grid}\n{len(grid)}")

# Find unique combinations
unique_combinations = np.unique(grid, axis=0)

print(f"\n\nUnique combinations (should be 216 in total): \n{unique_combinations}\nIts size: {len(unique_combinations)}")
