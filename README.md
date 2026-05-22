# Neurotransmitter Prediction from Electron Microscopy Images in BANC

[![DOI](https://zenodo.org/badge/1028540389.svg)](https://zenodo.org/badge/latestdoi/1028540389)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository is based off previous work from the preprint [Neurotransmitter Classification from
Electron Microscopy Images at Synaptic Sites in
Drosophila.](https://www.biorxiv.org/content/10.1101/2020.06.12.148775v2)

## Access to Neurotransmitter Predictions

Predictions for the neurotransmitters (GABA, acetylcholine,
glutamate, serotonin, octopamine, dopamine, histamine, and tyramine) on the BANC dataset are publicly available here:

Note: Only synapses detected with a size > 5 have NTs predicted.

`gs://leelab_fly_cns/files/banc_nt_prediction_w_sizethresh_5_09072025.parquet`

A simplied version of the neurotransmitter predictions for BANC on CAVE are located here:
https://cave.fanc-fly.com/annotation/views/aligned_volume/brain_and_nerve_cord/table/synapses_250226_nt_prediction_5


## Ground-Truth

The ground-truth train and test split used for BANC is located here:

`gs://leelab_fly_cns/files/banc_nt_ground_truth`