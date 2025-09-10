# mef-agri

evaluation framework for agricultural models

## documentation

https://josephinum-research.github.io/mef-agri/

## initial tasks to set up repository

- [ ] data sources

  - [ ] custom project-class for data access and 
  - [ ] INCA weather
  - [ ] eBod soil
  - [ ] Sentinel-2 planetary computer

- [ ] model definitions `mef_agri.models`

  - [x] SWAT atmosphere
  - [x] SWAT soil
  - [x] EPIC crop
  - [x] custom management
  - [x] INRAE Sentinel-2

- [ ] model evaluation

  - [x] statistical utilities for evaluation `mef_agri.evaluation.stats_utils.py`
  - [ ] database
  - [ ] model input
  - [x] base-class for estimators
  - [ ] estimators (model propagation, bootstrap particle filter)

- [ ] utils

- [x] sphinx documentation
