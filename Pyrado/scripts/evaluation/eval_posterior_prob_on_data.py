# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Script to evaluate a selection of posteriors on another set of rollouts, e.g..
"""
import os.path as osp
import pickle
import sys

import torch as to
from matplotlib import pyplot as plt
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.sbi_rollout_sampler import RecRolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


def _init(G, algo, dp_mapping, num_samples, normalize, use_mcmc):
    G.algo = pickle.loads(algo)
    G.dp_mapping = pickle.loads(dp_mapping)
    G.num_samples = pickle.loads(num_samples)
    G.normalize = pickle.loads(normalize)
    G.use_mcmc = pickle.loads(use_mcmc)


def _eval(G, posterior):
    _, log_probs_ml = G.algo.get_ml_posterior_samples(
        G.dp_mapping,
        posterior,
        data_eval,
        num_eval_samples=G.num_samples,
        num_ml_samples=1,
        calculate_log_probs=True,
        normalize_posterior=G.normalize,
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=G.use_mcmc),
    )
    return log_probs_ml


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)

    # Check the args
    if args.dir is None:
        raise pyrado.ValueErr(msg="Provide the path to the new experiment directory using --dir!")
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Load the algorithm
    algo = Algorithm.load_snapshot(
        osp.join(pyrado.TEMP_DIR, "qq-su", "npdr_qq-sub", "2021-03-11_11-56-34--lensegs-10_seed-42")  # TODO
    )
    if not isinstance(algo, (NPDR, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=(NPDR, BayesSim))

    # Load the posteriors
    eval_dict = {
        "A": pyrado.load(
            None,
            "posterior",
            "pt",
            osp.join(pyrado.TEMP_DIR, "qq-su", "npdr_qq-sub", "2021-03-11_11-56-34--lensegs-10_seed-42"),
            meta_info=None,
        ),
        "B": pyrado.load(
            None,
            "posterior",
            "pt",
            osp.join(pyrado.TEMP_DIR, "qq-su", "npdr_qq-sub", "2021-03-11_11-56-34--lensegs-10_seed-42"),
            meta_info=None,
        ),
    }
    posteriors = list(eval_dict.values())

    # Get the reference rollouts in the shape required for conditioning on them
    rollout_worker = RecRolloutSamplerForSBI(
        args.dir, algo.embedding, algo.num_segments, algo.len_segments, rand_init_rollout=False
    )
    data_eval = to.cat(
        [rollout_worker()[0] for _ in range(1)], dim=0
    )  # only data needed TODO rollout_worker.num_rollouts

    # Do the evaluation
    pool = SamplerPool(num_threads=2)
    pool.invoke_all(
        _init,
        pickle.dumps(algo),
        pickle.dumps(algo.dp_mapping),
        pickle.dumps(args.num_samples),
        pickle.dumps(args.normalize),
        pickle.dumps(args.use_mcmc),
    )
    with tqdm(leave=False, file=sys.stdout, unit="posteriors", desc="Evaluating") as pb:
        log_probs_ml = pool.run_map(_eval, posteriors, pb)

    # Average over the evaluation samples
    print(len(log_probs_ml))
    breakpoint()
    avg_log_probs = to.mean(log_probs_ml, dim=1)
    normalize_str = "normalized" if args.normalize else "unnormalized"
    print_cbt(f"average {normalize_str} probability per rollout {to.exp(avg_log_probs).numpy()}", "g")
