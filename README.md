\# Explainable Longitudinal Deep Learning for Time-Dependent Alzheimer's Disease Progression Prediction from Serial 3D MRI



This repository contains the reproducible analysis code accompanying the study:



\*\*Explainable Longitudinal Deep Learning for Time-Dependent Alzheimer's Disease Progression Prediction from Serial 3D MRI\*\*



\*\*Author:\*\* Wandile Juddy Nhlapho  

\*\*ORCID:\*\* 0000-0002-8685-8395



\---



\## Overview



This study investigates time-dependent Alzheimer's disease progression prediction from longitudinal structural 3D MRI.



The study was designed around four complementary questions:



1\. Does longitudinal modelling provide a clear predictive advantage over using only the latest available MRI?

2\. Does the longitudinal CNN-Transformer use information from multiple historical MRI visits when generating progression-risk predictions?

3\. How much do continuous-time encoding, Transformer-based temporal modelling, and longitudinal imaging contribute to the behaviour of the full longitudinal model?

4\. Does the frozen longitudinal CNN-Transformer transport to an independent external cohort without adaptation?



These questions were addressed using separate but complementary benchmark, explainability, ablation, and external-validation analyses.



\---



\## Study Design



\### ADNI development cohort



The internal study used longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).



The locked analysis cohort contained:



\- 972 participants

\- 4,247 MRI examinations

\- 3,634 dynamic prediction landmarks

\- 326 participants with progression events

\- 646 censored participants



The participant-level split was fixed using seed 2026:



| Split | Participants | Landmarks |

|---|---:|---:|

| Training | 680 | 2,530 |

| Validation | 146 | 521 |

| Test | 146 | 583 |



The test cohort remained frozen until final evaluation.



\---



\## Dynamic Survival Prediction



Prediction was formulated using dynamic landmarks.



For each participant and prediction landmark, only MRI examinations available up to that landmark were used.



The discrete survival framework used five 12-month intervals:



\- 0-12 months

\- 12-24 months

\- 24-36 months

\- 36-48 months

\- 48-60 months



The primary prediction horizons were 12, 24, and 36 months.



The 48- and 60-month horizons were retained as secondary analyses.



\---



\## MRI Processing



MRI processing followed a deterministic workflow designed to preserve the locked experimental protocol.



The original ADNI image grid was approximately:



```text

197 x 233 x 189 voxels

