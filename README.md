\# Explainable Longitudinal Deep Learning for Alzheimer's Disease Progression



Reproducible research code accompanying the study:



\*\*Explainable Longitudinal Deep Learning for Time-Dependent Alzheimer's Disease Progression Prediction from Serial 3D MRI\*\*



Author: \*\*Wandile Juddy Nhlapho\*\*  

ORCID: 0000-0002-8685-8395



\## Overview



This repository contains the analysis code used to investigate time-dependent Alzheimer's disease progression prediction from longitudinal structural MRI.



The study contains two complementary experimental branches that should be interpreted separately:



1\. a controlled internal benchmark comparing B0, B1, and B2 architectures; and

2\. a longitudinal CNN-Transformer reference model (M0) used for explainability, component ablation, and independent external validation.



The repository contains analysis code only. Raw ADNI and OASIS-3 imaging or clinical data are not redistributed.



\## Study Design



\### Controlled benchmark: B0-B2



Three architectures were compared using the frozen ADNI experimental protocol:



\- \*\*B0:\*\* latest-MRI 3D CNN.

\- \*\*B1:\*\* longitudinal 3D CNN with continuous-time encoding and masked mean pooling.

\- \*\*B2:\*\* longitudinal 3D CNN with continuous-time encoding and GRU aggregation.



Model selection within this benchmark was performed using validation negative log-likelihood before frozen-test evaluation.



B0 was selected within the B0-B2 benchmark. This selection should not be interpreted as defining B0 as the M0 longitudinal reference architecture used in the separate explainability and external-validation analyses.



\### Longitudinal reference model: M0



M0 is the full longitudinal CNN-Transformer architecture:



```text

Serial 3D MRI

&#x20;     |

Shared 3D CNN encoder

&#x20;     |

Continuous-time encoding

&#x20;     |

3-layer Transformer

&#x20;     |

Longitudinal pooling

&#x20;     |

Discrete-time survival head

