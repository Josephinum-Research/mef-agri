# mef-agri

evaluation framework for agricultural models

## documentation

https://josephinum-research.github.io/mef-agri/

## initial tasks to set up repository

- [ ] data sources

  - [x] custom project-class for data access and 
  - [x] INCA weather
  - [x] eBod soil
  - [x] Sentinel-2 planetary computer
  - [ ] management data

- [x] model definitions `mef_agri.models`

  - [x] SWAT atmosphere
  - [x] SWAT soil
  - [x] EPIC crop
  - [x] custom management
  - [x] INRAE Sentinel-2
  - [x] conditions in SWAT model

- [ ] model evaluation

  - [x] statistical utilities for evaluation `mef_agri.evaluation.stats_utils.py`
  - [x] database
  - [ ] model input
  - [x] base-class for estimators
  - [x] estimators (model propagation, bootstrap particle filter)

- [x] utils

- [x] sphinx documentation
