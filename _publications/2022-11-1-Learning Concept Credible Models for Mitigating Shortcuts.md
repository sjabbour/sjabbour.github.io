---
title: "Learning Concept Credible Models for Mitigating Shortcuts"
collection: publications
permalink: /publication/CCM
excerpt: 'Models can exploit spurious correlations as shortcuts during training. We aim to learn robust and accurate models from biased training data using both known and unknown concepts.'
venue: 'NeurIPS 2022'
paperurl: 'https://openreview.net/pdf?id=yKYCwTvl8eU'
citation: 'Jiaxuan Wang, Sarah Jabbour, Maggie Makar, Michael W. Sjoding, Jenna Wiens. Learning Concept Credible Models for Mitigating Shortcuts. *NeurIPS*, November 2022.'
---
<img src="/images/ccm_overview.png" style="height: 150px; width:300px;" align=left /> During training, models can exploit spurious correlations as shortcuts, resulting
in poor generalization performance when shortcuts do not persist. In this work,
assuming access to a representation based on domain knowledge (i.e., known
concepts) that is invariant to shortcuts, we aim to learn robust and accurate models
from biased training data. In contrast to previous work, we do not rely solely on
known concepts, but allow the model to also learn unknown concepts. We propose
two approaches for mitigating shortcuts that incorporate domain knowledge, while
accounting for potentially important yet unknown concepts. The first approach
is two-staged. After fitting a model using known concepts, it accounts for the
residual using unknown concepts. While flexible, we show that this approach
is vulnerable when shortcuts are correlated with the unknown concepts. This
limitation is addressed by our second approach that extends a recently proposed
regularization penalty. Applied to two real-world datasets, we demonstrate that
both approaches can successfully mitigate shortcut learning.


