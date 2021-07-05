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
Train agents to solve the Ball-on-Beam environment using Stein Variational Policy Gradient.
"""
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat


if __name__ == "__main__":
    dt = 1e-2
    env = BallOnBeamSim(dt, 500)

    ex_dir = setup_experiment(BallOnBeamSim.name, SVPG.name)

    hparam = {
        "particle_hparam": {
            # "actor": {"hidden_sizes": [32, 24], "hidden_nonlin": to.relu},
            "actor": dict(feats=FeatureStack(identity_feat, sin_feat)),
            "vfcn": {"hidden_sizes": [32, 32], "hidden_nonlin": to.relu},
            "critic": dict(
                gamma=0.99,
                lamda=0.95,
                batch_size=100,
                standardize_adv=True,
                lr_scheduler=lr_scheduler.ExponentialLR,
                lr_scheduler_hparam=dict(gamma=0.99),
            ),
            "algo": dict(
                max_iter=500,
                min_steps=env.max_steps * 10,
                num_workers=4,
                vfcn_coeff=0.7,
                entropy_coeff=4e-5,
                batch_size=256,
                std_init=0.8,
                lr=2e-3,
                lr_scheduler=lr_scheduler.ExponentialLR,
                lr_scheduler_hparam=dict(gamma=0.99),
            ),
        },
        "max_iter": 500,
        "num_particles": 4,
        "temperature": 0.001,
        "horizon": 20,
        "min_steps": env.max_steps * 10,
    }

    algo = SVPG(ex_dir, env, **hparam)

    algo.train()
