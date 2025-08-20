# MOBO with decoupled evaluations -
# Using Hypervolume Knowledge Gradient


import os

import torch

import pymoo
from pymoo.problems import get_problem
from scipy.stats import qmc
import numpy as np


def getPyMooProblem(function, n_var, n_obj):

    problem = get_problem(function, n_var=n_var, n_obj=n_obj)

    bl = problem.xl
    bu = problem.xu
    bounds = []

    for i in range(n_var):
        bounds.append([bl[i], bu[i]])

    return problem, np.array(bounds)


def evalPyMooProblem(function, vec):

    result = function.evaluate(vec)
    # result = np.append(result, [0])

    return result * -1


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

print("Using device:", tkwargs["device"])
print("Torch version", torch.__version__)

# this example optimises ZDT2, 2 objectives and 6 dimensions
# the objectives have heterogenous costs of 3 and 1 respectively

from botorch.test_functions.multi_objective import ZDT2
from botorch.models.cost import FixedCostModel

# problem = ZDT2(negate=True, dim=6).to(**tkwargs)
# print("Problem bounds:", problem.bounds)
# # print shape of bounds
# bounds = problem.bounds
# print("Problem bounds shape:", bounds.shape)

# define the cost model
# TODO these costs will need to be changed when I set this up for HydroShield
objective_costs = {0: 1.0, 1: 1.0}
objective_indices = list(objective_costs.keys())
objective_costs = {int(k): v for k, v in objective_costs.items()}
objective_costs_t = torch.tensor(
    [objective_costs[k] for k in sorted(objective_costs.keys())], **tkwargs
)
cost_model = FixedCostModel(fixed_cost=objective_costs_t)


# use a list of single task gps to model each objective

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel

# generating the initial training data - i can replace this with LHS generation
problem, bounds = getPyMooProblem("dtlz2", n_var=6, n_obj=2)
bounds = torch.from_numpy(bounds)
print("Problem bounds:", bounds)
# print shape of bounds
print("Problem bounds shape:", bounds.shape)

# change bounds from shape (6, 2) to (2, 6)
bounds_reversed = bounds.T
print("Reversed bounds shape:", bounds_reversed.shape)


def generate_initial_data(n):
    # generate training data

    initSampleSize = n
    # bounds = np.array(value)
    lowBounds = bounds[:, 0]
    highBounds = bounds[:, 1]

    # Generate one Latin Hypercube Sample (LHS) for each test function,
    # to be used for all optimisers/scalarisers using a population size of 20
    sampler = qmc.LatinHypercube(
        d=bounds.shape[0]
    )  # Dimension is determined from bounds
    sample = sampler.random(n=initSampleSize)
    train_x = qmc.scale(sample, lowBounds, highBounds)

    # Check for and systematically replace NaN values in initial population
    # Requires evaluating initial population
    train_obj_true = np.empty((0, 2))  # Assuming 2 objectives

    for i in range(initSampleSize):

        newObjvTgt = evalPyMooProblem(problem, train_x[i, :])

        train_obj_true = np.vstack((train_obj_true, newObjvTgt))

    print("Initial Population:")
    print(train_x)
    print("initial targets:\n", train_obj_true)

    train_obj_true = torch.from_numpy(train_obj_true)
    train_x = torch.from_numpy(train_x)

    # train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    # train_obj_true = problem(train_x)
    return train_x, train_obj_true


def initialize_model(train_x_list, train_obj_list):
    # define models for objective and constraint
    # print(bounds)
    # print(bounds.shape)
    # print(train_x_list[0].shape)
    # # reshape bounds to match the shape of train_x
    # bounds = bounds.reshape(1, -1)

    # print(problem.bounds)
    # print(problem.bounds.shape)
    train_x_list = [normalize(train_x, bounds_reversed) for train_x in train_x_list]
    models = []
    for i in range(len(train_obj_list)):
        train_y = train_obj_list[i]
        train_yvar = torch.full_like(train_y, 1e-7)  # noiseless
        models.append(
            SingleTaskGP(
                train_X=train_x_list[i],
                train_Y=train_y,
                train_Yvar=train_yvar,
                outcome_transform=Standardize(m=1),
                covar_module=ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=train_x_list[0].shape[-1],
                        lengthscale_prior=GammaPrior(2.0, 2.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
            )
        )
    model = ModelListGP(*models)
    # print(model)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


# i believe this helper function is for the non-decoupled case, so I will not be using it

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf


BATCH_SIZE = 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4

standard_bounds = torch.zeros(2, len(bounds), **tkwargs)
standard_bounds[1] = 1


# def optimize_qnehvi_and_get_observation(model, train_x, sampler):
#     """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
#     # partition non-dominated space into disjoint rectangles
#     acq_func = qLogNoisyExpectedHypervolumeImprovement(
#         model=model,
#         ref_point=problem.ref_point.tolist(),  # use known reference point
#         X_baseline=normalize(train_x, problem.bounds),
#         prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
#         sampler=sampler,
#     )
#     # optimize
#     candidates, _ = optimize_acqf(
#         acq_function=acq_func,
#         bounds=standard_bounds,
#         q=BATCH_SIZE,
#         num_restarts=NUM_RESTARTS,
#         raw_samples=RAW_SAMPLES,  # used for intialization heuristic
#         options={"batch_limit": 5, "maxiter": 200},
#         sequential=True,
#     )
#     # observe new values
#     new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
#     new_obj_true = problem(new_x)
#     return new_x, new_obj_true


"""
Below we define the following helper functions:

get_current_value for computing the current hypervolume of the hypervolume maximizing 
    set under the posterior mean.

optimize_HVKG_and_get_obs_decoupled to initialize and optimize HVKG to determine which 
    design to evaluate and which objective to evaluate the design on. This method obtains 
    the observation corresponding to that design.

"""

from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    _get_hv_value_function,
    qHypervolumeKnowledgeGradient,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import IIDNormalSampler

NUM_PARETO = 2 if SMOKE_TEST else 10
NUM_FANTASIES = 2 if SMOKE_TEST else 8
NUM_HVKG_RESTARTS = 1


def get_current_value(
    model,
    ref_point,
    bounds,
):
    """Helper to get the hypervolume of the current hypervolume
    maximizing set.
    """
    curr_val_acqf = _get_hv_value_function(
        model=model,
        ref_point=ref_point,
        use_posterior_mean=True,
    )
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds,
        q=NUM_PARETO,
        num_restarts=20,
        raw_samples=1024,
        return_best_only=True,
        options={"batch_limit": 5},
    )
    return current_value


def optimize_HVKG_and_get_obs_decoupled(model):
    """Utility to initialize and optimize HVKG."""
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    current_value = get_current_value(
        model=model,
        ref_point=torch.from_numpy(
            np.array((-1.75, -1.75))
        ),  # use known reference point
        bounds=standard_bounds,
    )

    acq_func = qHypervolumeKnowledgeGradient(
        model=model,
        ref_point=torch.from_numpy(
            np.array((-1.75, -1.75))
        ),  # use known reference point
        num_fantasies=NUM_FANTASIES,
        num_pareto=NUM_PARETO,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
    )

    # optimize acquisition functions and get new observations
    objective_vals = []
    objective_candidates = []
    for objective_idx in objective_indices:
        # set evaluation index to only condition on one objective
        # this could be multiple objectives
        X_evaluation_mask = torch.zeros(
            1,
            len(objective_indices),
            dtype=torch.bool,
            device=standard_bounds.device,
        )
        X_evaluation_mask[0, objective_idx] = 1
        acq_func.X_evaluation_mask = X_evaluation_mask
        candidates, vals = optimize_acqf(
            acq_function=acq_func,
            num_restarts=NUM_HVKG_RESTARTS,
            raw_samples=RAW_SAMPLES,
            bounds=standard_bounds,
            q=BATCH_SIZE,
            sequential=False,
            options={"batch_limit": 5},
        )
        objective_vals.append(vals.view(-1))
        objective_candidates.append(candidates)
    best_objective_index = torch.cat(objective_vals, dim=-1).argmax().item()
    eval_objective_indices = [best_objective_index]
    print(", Evaluated Objective = ", eval_objective_indices)
    candidates = objective_candidates[best_objective_index]
    vals = objective_vals[best_objective_index]
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=bounds_reversed)
    new_obj = evalPyMooProblem(problem, new_x.cpu().numpy())
    new_obj = torch.from_numpy(new_obj).to(**tkwargs)
    new_obj = new_obj[..., eval_objective_indices]
    return new_x, new_obj, eval_objective_indices


# define function to find model-estimated pareto set of
# designs under posterior mean using NSGA-II

# this is just to compare the estimated HV in each iteration to an analytical pareto front
# to compare regrets between optimisers.


import numpy as np
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import _is_non_dominated_loop
from gpytorch import settings

# try:
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# from pymoo.util.termination.max_gen import MaximumGenerationTermination


def get_model_identified_hv_maximizing_set(
    model,
    population_size=50,
    max_gen=50,
):
    """Optimize the posterior mean using NSGA-II."""
    # tkwargs = {
    #     "dtype": problem.ref_point.dtype,
    #     "device": problem.ref_point.device,
    # }
    dim = len(bounds)
    # since its bounds for each feature this gives the dimensionality of the feature landscape

    class PosteriorMeanPymooProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=dim,
                n_obj=2,
                type_var=np.double,
            )
            self.xl = np.zeros(dim)
            self.xu = np.ones(dim)

        def _evaluate(self, x, out, *args, **kwargs):
            X = torch.from_numpy(x).to(**tkwargs)
            is_fantasy_model = (
                isinstance(model, ModelListGP)
                and model.models[0].train_targets.ndim > 2
            ) or (not isinstance(model, ModelListGP) and model.train_targets.ndim > 2)
            with torch.no_grad():
                with settings.cholesky_max_tries(9):
                    # eval in batch mode
                    y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                    var = model.posterior(X.unsqueeze(-2)).variance.squeeze(-2)
                    std = var.sqrt()
                if is_fantasy_model:
                    y = y.mean(dim=-2)
                    std = std.mean(dim=-2)
            out["F"] = -y.cpu().numpy()
            out["uncertainty"] = std.cpu().numpy()  # stores the predictive uncertainty

    pymoo_problem = PosteriorMeanPymooProblem()
    algorithm = NSGA2(
        pop_size=population_size,
        eliminate_duplicates=True,
    )
    res = minimize(
        pymoo_problem,
        algorithm,
        termination=("n_gen", max_gen),
        # seed=0,  # fix seed
        verbose=False,
    )

    X = torch.tensor(
        res.X,
        **tkwargs,
    )
    X = unnormalize(X, bounds_reversed)
    # print(X, X.shape)
    # Y = problem(X)
    Y = torch.Tensor(res.F)

    std = torch.Tensor(res.pop.get("uncertainty"))
    # print("std shape:", std.shape)
    # print(Y, Y.shape)
    # compute HV
    partitioning = FastNondominatedPartitioning(
        ref_point=torch.from_numpy(np.array((-1.75, -1.75))), Y=Y
    )
    return partitioning.compute_hypervolume().item(), X, Y, std


# except ImportError:
#     NUM_DISCRETE_POINTS = 100 if SMOKE_TEST else 100000
#     CHUNK_SIZE = 512

#     def get_model_identified_hv_maximizing_set(
#         model,
#     ):
#         """Optimize the posterior mean over a discrete set."""
#         tkwargs = {
#             "dtype": problem.ref_point.dtype,
#             "device": problem.ref_point.device,
#         }
#         dim = problem.dim

#         discrete_set = torch.rand(NUM_DISCRETE_POINTS, dim, **tkwargs)
#         with torch.no_grad():
#             preds_list = []
#             for start in range(0, NUM_DISCRETE_POINTS, CHUNK_SIZE):
#                 preds = model.posterior(
#                     discrete_set[start : start + CHUNK_SIZE].unsqueeze(-2)
#                 ).mean.squeeze(-2)
#                 preds_list.append(preds)
#             preds = torch.cat(preds_list, dim=0)
#             pareto_mask = _is_non_dominated_loop(preds)
#             pareto_X = discrete_set[pareto_mask]
#         pareto_X = unnormalize(pareto_X, problem.bounds)
#         Y = problem(pareto_X)
#         # compute HV
#         partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point, Y=Y)
#         return partitioning.compute_hypervolume().item()


"""
comparing BO using HVKG vs non-decoupled qNEHVI

The Bayesian optimization "loop" for a batch size of 1 simply iterates the following steps:

given a surrogate model, choose a candidate design and objective to evaluate (for methods that leverage decoupled evaluations).
observe one or more objectives for the candidate design.
update the surrogate model.

"""

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MC_SAMPLES = 128 if not SMOKE_TEST else 16
COST_BUDGET = 110 if not SMOKE_TEST else 54
torch.manual_seed(0)
verbose = True
N_INIT = 2 * len(bounds) + 1

# total_cost = {"hvkg": 0.0, "qnehvi": 0.0, "random": 0.0}
total_cost = {"hvkg": 0.0}


# call helper functions to generate initial training data and initialize model
train_x_hvkg, train_obj_hvkg = generate_initial_data(n=N_INIT)
train_obj_hvkg_list = list(train_obj_hvkg.split(1, dim=-1))
print(train_obj_hvkg_list)
train_x_hvkg_list = [train_x_hvkg] * len(train_obj_hvkg_list)
mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list)
# train_obj_random_list = train_obj_hvkg_list
# train_x_random_list = train_x_hvkg_list
# train_x_qnehvi_list, train_obj_qnehvi_list = (
#     train_x_hvkg_list,
#     train_obj_hvkg_list,
# )
cost_hvkg = cost_model(train_x_hvkg).sum(dim=-1)
total_cost["hvkg"] += cost_hvkg.sum().item()
# cost_qnehvi = cost_hvkg
# cost_random = cost_hvkg
# total_cost["qnehvi"] = total_cost["hvkg"]
# total_cost["random"] = total_cost["hvkg"]
# mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi_list, train_obj_qnehvi_list)
# mll_random, model_random = initialize_model(train_x_random_list, train_obj_random_list)
# fit the models
fit_gpytorch_mll(mll_hvkg)
# fit_gpytorch_mll(mll_qnehvi)
# fit_gpytorch_mll(mll_random)

iteration = 0

# compute hypervolume
hv, features, targets, stddv = get_model_identified_hv_maximizing_set(model=model_hvkg)

np.savetxt(
    f"modelParetoFronts/features/featuresIter{iteration}.txt",
    torch.Tensor.numpy(features),
)
np.savetxt(
    f"modelParetoFronts/targets/targetsIter{iteration}.txt", torch.Tensor.numpy(targets)
)
np.savetxt(
    f"modelParetoFronts/uncertainties/stdIter{iteration}.txt", torch.Tensor.numpy(stddv)
)


hvs_hvkg = [hv]
if verbose:
    print(
        f"\nInitial: Hypervolume (qHVKG) = " f"({hvs_hvkg[-1]:>4.2f}).\n",
        end="",
    )
# run N_BATCH rounds of BayesOpt after the initial random batch
active_algos = {k for k, v in total_cost.items() if v < COST_BUDGET}
while any(v < COST_BUDGET for v in total_cost.values()):

    t0 = time.monotonic()
    if "hvkg" in active_algos:
        # generate candidates
        (
            new_x_hvkg,
            new_obj_hvkg,
            eval_objective_indices_hvkg,
        ) = optimize_HVKG_and_get_obs_decoupled(
            model_hvkg,
        )
        # print("eval objectives: ", eval_objective_indices_hvkg)
        # update training points
        for i in eval_objective_indices_hvkg:
            train_x_hvkg_list[i] = torch.cat([train_x_hvkg_list[i], new_x_hvkg])
            train_obj_hvkg_list[i] = torch.cat(
                [train_obj_hvkg_list[i], new_obj_hvkg], dim=0
            )
        # print(train_obj_hvkg_list[0].shape)
        # print(train_obj_hvkg_list[1].shape)
        # update costs
        all_outcome_cost = cost_model(new_x_hvkg)
        new_cost_hvkg = all_outcome_cost[..., eval_objective_indices_hvkg].sum(dim=-1)
        cost_hvkg = torch.cat([cost_hvkg, new_cost_hvkg], dim=0)
        total_cost["hvkg"] += new_cost_hvkg.sum().item()
        # fit models
        mll_hvkg, model_hvkg = initialize_model(train_x_hvkg_list, train_obj_hvkg_list)
        fit_gpytorch_mll(mll_hvkg)

    # if "qnehvi" in active_algos:
    #     qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
    #     # generate candidates
    #     new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(
    #         model_qnehvi, train_x_qnehvi_list[0], qnehvi_sampler
    #     )
    #     # update training points
    #     for i in objective_indices:
    #         train_x_qnehvi_list[i] = torch.cat([train_x_qnehvi_list[i], new_x_qnehvi])
    #         train_obj_qnehvi_list[i] = torch.cat(
    #             [train_obj_qnehvi_list[i], new_obj_qnehvi[..., i : i + 1]]
    #         )
    #     # update costs
    #     new_cost_qnehvi = cost_model(new_x_qnehvi).sum(dim=-1)
    #     cost_qnehvi = torch.cat([cost_qnehvi, new_cost_qnehvi], dim=0)
    #     total_cost["qnehvi"] += new_cost_qnehvi.sum().item()
    #     # fit models
    #     mll_qnehvi, model_qnehvi = initialize_model(
    #         train_x_qnehvi_list, train_obj_qnehvi_list
    #     )
    #     fit_gpytorch_mll(mll_qnehvi)
    # if "random" in active_algos:
    #     # generate candidates
    #     new_x_random, new_obj_random = generate_initial_data(n=BATCH_SIZE)
    #     # update training points
    #     for i in objective_indices:
    #         train_x_random_list[i] = torch.cat([train_x_random_list[i], new_x_random])
    #         train_obj_random_list[i] = torch.cat(
    #             [train_obj_random_list[i], new_obj_random[..., i : i + 1]]
    #         )
    #     # update costs
    #     new_cost_random = cost_model(new_x_random).sum(dim=-1)
    #     cost_random = torch.cat([cost_random, new_cost_random], dim=0)
    #     total_cost["random"] += new_cost_random.sum().item()
    #     # fit models
    #     mll_random, model_random = initialize_model(
    #         train_x_random_list, train_obj_random_list
    #     )
    #     fit_gpytorch_mll(mll_random)

    # compute hypervolume
    for label, model, hv_list in zip(
        ["hvkg"],
        [model_hvkg],
        [hvs_hvkg],
    ):
        if label in active_algos:
            hv, features, targets, stddv = get_model_identified_hv_maximizing_set(
                model=model
            )
            hv_list.append(hv)
        else:
            # no update performed
            hv_list.append(hv_list[-1])

    t1 = time.monotonic()
    if verbose:
        print(
            f"\nBatch {iteration:>2}: Costs (qHVKG) = "
            f"({total_cost['hvkg']:>4.2f}). "
        )
        print(
            f"\nHypervolume (qHVKG) = ",
            f"({hvs_hvkg[-1]:>4.2f}), ",
            f"time = {t1-t0:>4.2f}.",
            end="",
        )
    else:
        print(".", end="")

    # for each list in train_objv_hvkg_list, save the list as a text file
    for i, train_objv_hvkg in enumerate(train_obj_hvkg_list):
        np.savetxt(
            f"objtv{i}/train_obj_hvkg_{iteration}.txt",
            train_objv_hvkg.cpu().numpy(),
            delimiter=",",
        )

    iteration += 1
    np.savetxt(
        f"modelParetoFronts/features/featuresIter{iteration}.txt",
        torch.Tensor.numpy(features),
    )
    np.savetxt(
        f"modelParetoFronts/targets/targetsIter{iteration}.txt",
        torch.Tensor.numpy(targets),
    )
    np.savetxt(
        f"modelParetoFronts/uncertainties/stdIter{iteration}.txt",
        torch.Tensor.numpy(stddv),
    )

    active_algos = {k for k, v in total_cost.items() if v < COST_BUDGET}
