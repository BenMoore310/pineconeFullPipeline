import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=20)
pf = get_problem("dtlz2").pareto_front(ref_dirs)

#save the pareto front as a numpy array and plot it
pfArray = np.array(pf)

# read in each iteration's features, targets, and std

filesList = glob.glob('modelParetoFronts/features/*')


for iteration in range(len(filesList)):  
    features = np.loadtxt(f"modelParetoFronts/features/featuresIter{iteration}.txt")
    targets = np.loadtxt(f"modelParetoFronts/targets/targetsIter{iteration}.txt")
    std = np.loadtxt(f"modelParetoFronts/uncertainties/stdIter{iteration}.txt")

# plot targets with error bars
    plt.figure(figsize=(10, 10))
    plt.errorbar(targets[:, 0], targets[:, 1], xerr=std[:,0], yerr=std[:, 1], fmt='o', label='Objective 1', capsize=5, errorevery=1)
    plt.scatter(pfArray[:, 0], pfArray[:, 1], c='black', marker='x', label='Pareto Front')
    plt.title(f'Iteration {iteration} - Targets with Uncertainty')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.xlim(-0.2, 1.5)
    plt.ylim(-0.2, 1.5)
    # plt.grid(True)
    plt.savefig(f"modelParetoFronts/paretoPlots/iteration_{iteration}.png")
    plt.close()
    # plt.show()

# load in all saved figures and generate an animated GIF
import imageio
images = []
for iteration in range(len(filesList)):
    images.append(imageio.imread(f"modelParetoFronts/paretoPlots/iteration_{iteration}.png"))
imageio.mimsave('pareto_animation.gif', images, duration=1)   
# print completion message
print("Plots generated and saved as GIF.")
