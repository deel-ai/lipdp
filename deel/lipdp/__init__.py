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
from os import path

with open(path.join(path.dirname(__file__), "VERSION")) as f:
    __version__ = f.read().strip()

from deel.lipdp.layers import (
    DP_ScaledL2NormPooling2D,
    DP_Flatten,
    DP_SpectralConv2D,
    DP_SpectralDense,
    DP_QuickFrobeniusDense,
    DP_Reshape,
    DP_Lambda,
    DP_Permute,
    DP_Reshape,
    DP_GroupSort,
    DP_InputLayer,
    DP_BoundedInput,
    DP_Flatten,
    DP_ClipGradient,
    DP_AddBias,
    DP_LayerCentering,
    DPLayer,
    make_residuals,
)
from deel.lipdp.losses import (
    DP_KCosineSimilarity,
    DP_MeanAbsoluteError,
    DP_MulticlassHinge,
    DP_MulticlassHKR,
    DP_MulticlassKR,
    DP_TauCategoricalCrossentropy,
)
from deel.lipdp.accounting import DPGD_Mechanism
from deel.lipdp.dynamic import LaplaceAdaptiveLossGradientClipping
from deel.lipdp.model import (
    DP_Model,
    DP_Sequential,
    DP_Accountant,
)
from deel.lipdp.pipeline import (
    load_adbench_data,
    prepare_tabular_data,
    load_and_prepare_images_data,
    bound_clip_value,
    bound_normalize,
)
from deel.lipdp.sensitivity import (
    get_max_epochs,
)
from deel.lipdp.utils import (
    CertifiableAUROC,
    PrivacyMetrics,
    ScaledAUC,
    SignaltoNoiseAverage,
    SignaltoNoiseHistogram,
)
