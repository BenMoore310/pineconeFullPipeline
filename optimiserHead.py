import numpy as np
from scipy.stats import qmc
import argparse
import validCurveGenerator as VCG
import matplotlib.pyplot as plt
import json
import subprocess
import scienceplots
plt.style.use(['science', 'notebook'])

cwdPath = '/home/bm424/Projects/pineconeFullPipeline'


def simulateDesign(objective):
    """
    Simulate the design using the current parameters.
    """

    objective_strings = ['025', '05', '075', '100', '125', '150']
    directoryString = objective_strings[objective]

    src = f'0_timesteps/0.{directoryString}'
    dst = './0'

    cloudSrc = f'cloudPropertiesFiles/kinematicCloudProperties{directoryString}'
    cloudDst = 'constant/kinematicCloudProperties'

    subprocess.run(['cp', '-r', src, dst], cwd=cwdPath + '/runDirectory', check=True)
    
    subprocess.run(['cp', cloudSrc, cloudDst], cwd=cwdPath + '/runDirectory', check=True)
    
    # Placeholder for simulation logic
    print("Simulating design with current parameters...")


    subprocess.run(["bash", "allRun"], cwd = cwdPath + '/runDirectory', check=True)

    subprocess.run(["bash", "calculateDesignEfficiency"], cwd = cwdPath + '/runDirectory', check=True)

    efficiencies = np.loadtxt('efficiency.txt')

    numberEfficiency = efficiencies[0]
    massEfficiency = efficiencies[1]

    subprocess.run(['rm', '-r', 0], cwd=cwdPath + '/runDirectory', check=True)


    return numberEfficiency, massEfficiency


    


def betaParamsToJSON(sample, numBasis):
    structured = []
    if numBasis > 1:
        for i in range(numBasis):
            idx = i * 3
            entry = {
                "basis": [sample[idx], sample[idx + 1]],
                "weight": [sample[idx + 2]]
            }
            structured.append(entry)
    else:
        entry = {
            "basis": [sample[0], sample[1]],
            "weight": [1.0]  # If only one basis, weight is set to 1.0
        }
        structured.append(entry)
    
    with open('trialParams.json', 'w') as f:
        json.dump(structured, f, indent=4)


def main(numBasis, numObj, initialSamples, weightingType, seed):

    # list of arrays of values for each objective
    objectiveList = [np.array([]) for _ in range(numObj)]

    evaluatedObjectives = np.random.randint(low=0, 
                                            high=6,
                                            size = (initialSamples),
                                            dtype=int )

    lbBasis = 0.0
    ubBasis = 12.0

    lbWeight = 0.0
    ubWeight = 1.0

    bounds = []

    # need to set it so that if numBasis= 1 then no weights are used
    if numBasis == 1:
        bounds.append([lbBasis, ubBasis])
        bounds.append([lbBasis, ubBasis])
    else:
        for i in range(numBasis):
            bounds.append([lbBasis, ubBasis])
            bounds.append([lbBasis, ubBasis])
            bounds.append([lbWeight, ubWeight])
    
    bounds = np.array(bounds)

    lowBounds = bounds[:, 0]
    highBounds = bounds[:, 1]

    # Generate Latin Hypercube samples
    #TODO add a check later on where if the number of bases is 1, then the weight parameter is set to 1.0
    if numBasis == 1:
        sampler = qmc.LatinHypercube(d=(numBasis*2), seed=seed)
    else:
        sampler = qmc.LatinHypercube(d=(numBasis*3), seed=seed)
 
    samples = sampler.random(n=initialSamples)

    # Scale samples to bounds
    initialPopulation = qmc.scale(samples, lowBounds, highBounds)

    print('initialPopulation', initialPopulation)

    for sample in initialPopulation:

        i = 0

        print('sample', sample)
        betaParamsToJSON(sample, numBasis)


        with open('trialParams.json', 'r') as f:
            data = json.load(f)

            paramArray = []
            weightArray = []
            for entry in data:
                paramArray.extend(entry['basis'])
                weightArray.extend(entry['weight'])
            
        paramArray = np.array(paramArray).reshape(-1, 2)
        weightArray = np.array(weightArray)
        print(paramArray.shape)
        print(weightArray.shape)


        # the first check for validity occurs here - by checking the distance
        # between plates for the given beta-CDF parameters
        # 
        VCG.main(numBasis, 'trialParams.json', 'random', resolution=50)
        
        currentSpline = np.loadtxt('runDirectory/spline.txt')
        # plot the current spline and save to a figure
        plt.figure()
        plt.plot(currentSpline[:, 0], currentSpline[:, 1], label='Current Spline')
        plt.title('Current Evaluated Plate Shape')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('currentSpline.png')
        plt.close()

        numberEfficiency, massEfficiency = simulateDesign(evaluatedObjectives[i])


        i += 1
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the head script for HydroShield optimisation.")
    parser.add_argument(
        "--numBasis",
        type=int,
        default=2,
        help="Number of Beta basis functions to use in the curve generation."
    )
    parser.add_argument(
        "--numObj",
        type=int,
        default=1,
        help="Number of objectives to be optimised over."
    )
    parser.add_argument(
        "--initialSamples",
        type=int,
        default=10,
        help="Number of initial samples for the Latin Hypercube sampling."
    )

    parser.add_argument(
        "--weightingType",
        type=str,
        choices=['random', 'equal'],
        default='random',
        help="Type of weighting for the Beta basis functions."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    main(args.numBasis, args.numObj, args.initialSamples, args.weightingType, args.seed)
