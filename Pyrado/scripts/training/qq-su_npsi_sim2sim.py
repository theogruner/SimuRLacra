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
import os
import os.path as osp
from copy import deepcopy

import yaml
from sbi import utils
from sbi.inference import SNPE_C
from sbi.inference.snpe import SNPE_A

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import load_dict_from_yaml, save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import select_query
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--h_file",
        type=str,
        nargs="?",
        help="name of hyper-parameter file",
    )
    parser.add_argument(
        "--use_settings",
        action="store_true",
        default=False,
        help="use the hyper-parameter settings from the settings file instead of the hyperparams.yaml",
    )
    args = parser.parse_args()

    # import the general settings from the defined yaml file
    hparam_dict = osp.join(pyrado.HPARAM_DIR, "npdr", "qq-su")
    setting_args = load_dict_from_yaml(osp.join(hparam_dict, "settings_npsi.yaml"))
    dp_real = setting_args["dp_real"]

    if args.h_file is not None:
        hparam_file = args.h_file
    elif args.use_settings:
        hparam_file = "settings_npsi.yaml"
    else:
        hparam_files = []
        for file in os.listdir(hparam_dict):
            hparam_files.append(file)
        hparam_file = select_query(hparam_files)

    args = vars(args)

    # override the settings hyper-parameters with hparam-yaml file
    try:
        hparams_path = osp.join(hparam_dict, hparam_file)
        hparam_args = load_dict_from_yaml(hparams_path)
        setting_args.update(hparam_args)
    except pyrado.PathErr as e:
        print(e)

    args.update(setting_args)

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}", "sim2sim")

    # Set seed if desired
    pyrado.set_seed(args["seed"], verbose=True)

    # Environments
    env_sim = QQubeSwingUpSim(args["env_hparam"]["dt"], args["env_hparam"]["max_steps"])
    env_sim = ActDelayWrapper(env_sim, args["env_hparam"]["act_delay"])
    env_sim.domain_param = args["dp_real"]

    # Create a fake ground truth target domain
    env_real = deepcopy(env_sim)
    env_real.domain_param = dp_real
    dp_nom = env_sim.get_nominal_domain_param()

    # Behavioral policy
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **args["policy_hparam"])

    # define prior
    prior = utils.BoxUniform(**args["prior"])

    # Time series embedding
    embedding = create_embedding(args["embedding_name"], env_sim.spec, **args["embedding_hparam"])

    # define sbi subroutine. Choose between SNPE-A and SNPE-C
    if args["sbi_subrtn_name"] == "SNPE-A":
        sbi_subrtn = SNPE_A
    elif args["sbi_subrtn_name"] == "SNPE-C":
        sbi_subrtn = SNPE_C
    else:
        raise pyrado.ValueErr(given=args["sbi_subrtn_name"], eq_constraint="SNPE-A or SNPE-C")

    # Algorithm
    algo_hparam = dict(
        **args["npdr_hparam"],
        posterior_hparam=args["posterior_hparam"],
        subrtn_sbi_training_hparam=args["subrtn_sbi_training_hparam"],
        subrtn_sbi_sampling_hparam=args["subrtn_sbi_sampling_hparam"],
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        args["dp_mapping"],
        prior,
        sbi_subrtn,
        embedding,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(setting_args, save_dir=ex_dir)

    algo.train(seed=args["seed"])
