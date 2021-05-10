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
Domain parameter identification experiment on the Quanser Qube environment using Neural Posterior Domain Randomization
"""
import torch as to
from sbi import utils
from sbi.inference import SNPE_C
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    RNNEmbedding,
)
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str, num_segs_str, len_seg_str = "", "", ""
    if args.seed is not None:
        seed_str = f"_seed-{args.seed}"
    if args.num_segments is not None:
        num_segs_str = f"numsegs-{args.num_segments}"
    elif args.len_segments is not None:
        len_seg_str = f"lensegs-{args.len_segments}"
    else:
        raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")
    num_eval_samples = args.num_samples or 50

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name,
        f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}",
        num_segs_str + len_seg_str + seed_str,
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=int(5.5 * 250))
    env_sim = QQubeSwingUpSim(**env_sim_hparams)

    # Create the ground truth target domain and the behavioral policy
    num_real_rollouts = 1
    env_real = QQubeSwingUpReal(**env_sim_hparams)
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env_sim.spec, **policy_hparam)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km", 4: "Mr", 5: "Mp", 6: "Lr", 7: "Lp", 8: "g", 9: "act_delay"}

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        # low=to.tensor([-0.1, 0.0, 0]),
        # high=to.tensor([0.0, 0.1, 10])
        # low=to.tensor([0.0, 0.0]),
        # high=to.tensor([2 * 0.0015, 2 * 0.0005]),
        # low=to.tensor([dp_nom["Dr"] * 0, dp_nom["Dp"] * 0, dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5]),
        # high=to.tensor([dp_nom["Dr"] * 10, dp_nom["Dp"] * 10, dp_nom["Rm"] * 2.0, dp_nom["km"] * 2.0]),
        # low=to.tensor([dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5, dp_nom["Mr"] * 0.5, dp_nom["Mp"] * 0.5]),
        # high=to.tensor([dp_nom["Rm"] * 1.5, dp_nom["km"] * 1.5, dp_nom["Mr"] * 1.5, dp_nom["Mp"] * 1.5]),
        low=to.tensor(
            [
                dp_nom["Dr"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.1,
                dp_nom["km"] * 0.2,
                dp_nom["Mr"] * 0.3,
                dp_nom["Mp"] * 0.3,
                dp_nom["Lr"] * 0.5,
                dp_nom["Lp"] * 0.5,
                dp_nom["g"] * 0.85,
                # -0.1,
                # 0.0,
                0,
            ]
        ),
        high=to.tensor(
            [
                dp_nom["Dr"] * 5,
                dp_nom["Dp"] * 50,
                dp_nom["Rm"] * 1.9,
                dp_nom["km"] * 1.8,
                dp_nom["Mr"] * 1.7,
                dp_nom["Mp"] * 1.7,
                dp_nom["Lr"] * 1.5,
                dp_nom["Lp"] * 1.5,
                dp_nom["g"] * 1.15,
                # 0,
                # 0.1,
                5,
            ]
        ),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(
        downsampling_factor=1,
        # len_rollouts=env_sim.max_steps,
        # recurrent_network_type=nn.RNN,
        # only_last_output=True,
        # hidden_size=20,
        # num_recurrent_layers=1,
        # output_size=1,
    )
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=10)

    # Policy optimization subroutine
    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.relu)
    vfcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9844224855479998,
        lamda=0.9700148505302241,
        num_epoch=5,
        batch_size=500,
        standardize_adv=False,
        lr=7.058326426522811e-4,
        max_grad_norm=6.0,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    critic = GAE(vfcn, **critic_hparam)

    subrtn_policy_hparam = dict(
        max_iter=5,
        eps_clip=0.12648736789309026,
        min_steps=30 * env_sim.max_steps,
        num_epoch=7,
        batch_size=500,
        std_init=0.7573286998997557,
        lr=6.999956625305722e-04,
        max_grad_norm=1.0,
        num_workers=8,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    subrtn_policy = PPO(ex_dir, env_sim, policy, critic, **subrtn_policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=150,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=3000,
        num_sbi_rounds=5,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=num_eval_samples,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        use_rec_act=True,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=10,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        subrtn_policy_snapshot_mode="best",
        num_workers=args.num_workers,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        SNPE_C,
        embedding,
        **algo_hparam,
        subrtn_policy=subrtn_policy,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_sim_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(subrtn_policy=subrtn_policy_hparam, subrtn_policy_name=subrtn_policy.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
