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
from shutil import copyfile

import torch as to
import yaml
from sbi import utils
from sbi.inference import SNPE_C
from sbi.inference.snpe import SNPE_A

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import load_dict_from_yaml, save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    LastStepEmbedding,
    RNNEmbedding,
)
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import select_query
from pyrado.utils.sbi import create_embedding


# TODO: where is the best place to put this?
def update(base_dict, update_dict):
    """
    Updates dictionary with several levels of sub dictionaries.
    :param base_dict: dictionary to update
    :param update_dict: dictionary used for the update
    :return: the updated base_dict
    """
    for key, val in update_dict.items():
        if key in base_dict.keys():  # update only if key is in base_dict
            if isinstance(val, dict):
                base_dict[key] = update(base_dict.get(key, {}), val)  # iterate through several levels of dicts
            else:
                base_dict[key] = val  # update dict
    return base_dict


experiments = {
    "qq-su": QQubeSwingUpSim,
}
policies = {
    "qq-sub": QQubeSwingUpAndBalanceCtrl,
}

if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--h_file",
        type=str,
        nargs="?",
        help="name of hyper-parameter file",
    )

    args = parser.parse_args()
    args = vars(args)

    # import the general settings from the defined yaml file

    if args["h_file"] is not None:
        hparam_file = args["h_file"]
    else:
        hparam_files = []
        for file in os.listdir(pyrado.HPARAM_DIR):
            if file.endswith(".yaml") and file.startswith("hparam_"):
                hparam_files.append(file)
        if not hparam_files:
            raise FileNotFoundError()
        hparam_file = select_query(hparam_files)

    hparam_args = load_dict_from_yaml(osp.join(pyrado.HPARAM_DIR, hparam_file))
    settings_path = osp.join(
        pyrado.HPARAM_DIR,
        hparam_args["algo_name"],
        hparam_args["env_name"],
        "settings.yaml",
    )
    setting_args = load_dict_from_yaml(settings_path)

    # update settings_args, if identical hyperparams are defined in hyperparams_args
    update(setting_args, hparam_args)

    # ---- Environments
    base_env = experiments[setting_args["env_name"]]
    # create simulated environment
    env_sim = base_env(setting_args["env_hparam"]["dt"], setting_args["env_hparam"]["max_steps"])
    # TODO: Applying Wrappers by keywords from hparam_args
    # env_sim = ActDelayWrapper(env_sim, setting_args["env_hparam"]["act_delay"])
    env_sim = GaussianObsNoiseWrapper(env_sim, domain_param=setting_args["dp_real"])

    # create real environment
    env_real = deepcopy(env_sim)
    env_real.domain_param = setting_args["dp_real"]

    # ---- Update parameters
    # update domain parameter of simulated environment. This represents a change in the
    # nominal parameters and is used to define the prior range
    if "updated_nominals" in setting_args.keys() and isinstance(setting_args["updated_nominals"], dict):
        for dp in setting_args["updated_nominals"].keys():
            env_sim.domain_param[dp] = setting_args["updated_nominals"][dp]

    # update dp mapping and build prior
    prior_bounds = dict(low=[], high=[])
    dp_mapping = dict()
    dp_counter = 0
    for key in sorted(setting_args["dp_mapping"].keys()):
        dp = setting_args["dp_mapping"][key]
        if dp in hparam_args["dp_selection"]:
            prior_bounds["low"].append((1.0 - setting_args["prior_range"][dp]) * env_sim.domain_param[dp])
            prior_bounds["high"].append((1.0 + setting_args["prior_range"][dp]) * env_sim.domain_param[dp])
            dp_mapping[dp_counter] = dp
            dp_counter += 1

    # ---- Define objects for algo
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        setting_args["env_name"],
        f"{NPDR.name}_{setting_args['policy_name']}",
        "sim2sim",
    )
    # Set seed if desired
    pyrado.set_seed(setting_args["seed"], verbose=True)
    # Behavioral policy
    policy = policies[setting_args["policy_name"]](env_sim.spec, **setting_args["policy_hparam"])
    # define prior
    prior = utils.BoxUniform(**prior_bounds)
    # Time series embedding
    embedding = create_embedding(setting_args["embedding_name"], env_sim.spec, **setting_args["embedding_hparam"])

    # define sbi subroutine. Choose between SNPE-A and SNPE-C
    if setting_args["sbi_subrtn_name"] == "SNPE-A":
        sbi_subrtn = SNPE_A
    elif setting_args["sbi_subrtn_name"] == "SNPE-C":
        sbi_subrtn = SNPE_C
    else:
        raise pyrado.ValueErr(
            given=setting_args["sbi_subrtn_name"],
            msg="Currently only SNPE-C is supported",
        )

    # ---- Algorithm
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        sbi_subrtn,
        embedding,
        **setting_args["algo_hparam"],
    )

    # Save the hyper-parameters
    copyfile(osp.join(pyrado.HPARAM_DIR, hparam_file), osp.join(ex_dir, hparam_file))
    copyfile(settings_path, osp.join(ex_dir, "settings.yaml"))

    # Yeeehaaaa!
    algo.train(seed=args["seed"])
