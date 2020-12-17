import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
xm = 0.2
xa = 4.0
ym = 3.0
f = lambda x: ym * np.sin(np.pi * xm * (x[0] + xa)) + \
              ym * np.sin(2 * xm * np.pi * (x[0] + xa)) / 2.0 \
              + x[0] / 10.0

# Set data for model
data = Namespace()
#data.x = [[4.0]]
data.x = []
data.y = [f(x) for x in data.x]

# Set arrays
min_x = 3.5
max_x = 20.0
x_path_big = [[x] for x in np.linspace(3.8, 8.0, 200)]
x_test = [[x] for x in np.linspace(0.0, max_x+2, 550)]
y_test = [f(x) for x in x_test]

# Set "true execution path"
exe_path_true = Namespace(x=x_path_big, y=[f(x) for x in x_path_big])

# Plot setup
fig = plt.figure(figsize=(8, 5))
plt.xlim([0, max_x + 1])
plt.ylim([-7.0, 8.0])
plt.xlabel('x')
plt.ylabel('y')

# Plot true function
plt.plot(x_test, y_test, '-', color='k', linewidth=2)

# Show plot
neatplot.save_figure(f'gp_test_12')
plt.show()
