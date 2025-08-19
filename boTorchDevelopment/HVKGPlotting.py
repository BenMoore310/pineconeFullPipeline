import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
# read in each iteration's features, targets, and std

filesList = glob.glob('modelParetoFronts/features/*')


for iteration in range(len(filesList)):  
    features = np.loadtxt(f"modelParetoFronts/features/featuresIter{iteration}.txt")
    targets = np.loadtxt(f"modelParetoFronts/targets/targetsIter{iteration}.txt")
    std = np.loadtxt(f"modelParetoFronts/uncertainties/stdIter{iteration}.txt")

# plot targets with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(targets[:, 0], targets[:, 1], xerr=std[:,0], yerr=std[:, 1], fmt='o', label='Objective 1', capsize=5, errorevery=3)
    plt.title(f'Iteration {iteration} - Targets with Uncertainty')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.xlim(-0.2, 1.75)
    plt.ylim(-0.2, 1.75)
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
