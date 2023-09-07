# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from autodp import mechanism_zoo
from autodp import transformer_zoo
from autodp.autodp_core import Mechanism


class DPGD_Mechanism(Mechanism):
    """DPGD Mechanism.

    Args:
        mode (str): kind of mechanism to use. Either 'global' or "per-layer".
        prob (float): probability of subsampling, equal to batch_size / dataset_size.
        noise_multipliers (float, or list of floats): single scalar when mode == 'global', list of scalars when mode == "per-layer".
        num_grad_steps (int): number of gradient steps.
        delta (float): delta parameter for DP.
        dynamic_clipping (optional, dict): dictionary of parameters for dynamic clipping.
            Keys depend on the mode of dynamic clipping, but it always contains a "mode" key.
    """

    def __init__(
        self,
        mode,
        prob,
        noise_multipliers,
        num_grad_steps,
        delta,
        dynamic_clipping=None,
        name="DPGD_Mechanism",
    ):
        # Init
        Mechanism.__init__(self)
        self.name = name
        self.params = {
            "prob": prob,
            "noise_multipliers": noise_multipliers,
            "num_grad_steps": num_grad_steps,
            "delta": delta,
            "dynamic_clipping": dynamic_clipping,
        }

        assert mode in ["global", "per-layer"], "Unknown mode for DPGD_Mechanism."

        if mode == "global":
            model_mech = mechanism_zoo.GaussianMechanism(sigma=noise_multipliers)
            # assert model_mech.neighboring == "add_remove"
        elif mode == "per-layer":
            layer_mechanisms = []

            for sigma in noise_multipliers:
                mech = mechanism_zoo.GaussianMechanism(sigma=sigma)
                # assert mech.neighboring == "add_remove"
                layer_mechanisms.append(mech)

            # Accountant composition on layers
            compose_gaussians = transformer_zoo.ComposeGaussian()
            model_mech = compose_gaussians(
                layer_mechanisms, [1] * len(noise_multipliers)
            )

        subsample_grad_computation = transformer_zoo.AmplificationBySampling()
        sub_sampled_model_gaussian_mech = subsample_grad_computation(
            # improved_bound_flag can be set to True for Gaussian mechanisms (see autodp documentation).
            model_mech,
            prob,
            improved_bound_flag=True,
        )

        compose = transformer_zoo.Composition()
        mechs_to_compose = [sub_sampled_model_gaussian_mech]
        niter_to_compose = [num_grad_steps]

        if dynamic_clipping["mode"] == "laplace":
            # TODO: the pure DP mechanism should be sub-sampled to improve the bound.
            dynamic_clipping_mech = mechanism_zoo.PureDP_Mechanism(
                eps=dynamic_clipping["epsilon"], name="SVT"
            )
            mechs_to_compose.append(dynamic_clipping_mech)
            niter_to_compose.append(dynamic_clipping["num_updates"])
        elif dynamic_clipping["mode"] == "quantiles":
            private_quantiles_mech = mechanism_zoo.GaussianMechanism(
                sigma=dynamic_clipping["noise_multiplier"]
            )
            subsample_quantiles = transformer_zoo.AmplificationBySampling()
            subsampled_private_quantiles_mech = subsample_quantiles(
                private_quantiles_mech, prob, improved_bound_flag=True
            )
            mechs_to_compose.append(subsampled_private_quantiles_mech)
            niter_to_compose.append(dynamic_clipping["num_updates"])

        global_mech = compose(mechs_to_compose, niter_to_compose)

        # assert global_mech.neighboring in ["add_remove", "add_only", "remove_only"]

        # Get relevant information
        self.epsilon = global_mech.get_approxDP(delta=delta)
        self.delta = delta

        # Propagate updates
        rdp_global = global_mech.RenyiDP
        self.propagate_updates(rdp_global, type_of_update="RDP")
