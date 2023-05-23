# -*- coding: utf-8 -*-
# Copyright anonymized et anonymized - All
# rights reserved. anonymized is a research program operated by anonymized, anonymized,
# anonymized and anonymized - https://www.anonymized.ai/
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

from lipdp.losses import (
    DP_KCosineSimilarity,
    DP_MeanAbsoluteError,
    DP_MulticlassHinge,
    DP_MulticlassHKR,
    DP_MulticlassKR,
    DP_TauCategoricalCrossentropy,
)
from lipdp.model import DP_Model, DP_Sequential, DP_Accountant
from lipdp.pipeline import load_and_prepare_data, bound_clip_value, bound_normalize
from lipdp.sensitivity import (
    get_max_epochs,
    gradient_norm_check,
    check_layer_gradient_norm,
)