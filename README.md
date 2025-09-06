# mef-agri

evaluation framework for agricultural models

## initial tasks to set up repository

- [ ] data sources

  - [ ] custom project-class for data access and 
  - [ ] INCA weather
  - [ ] eBod soil
  - [ ] Sentinel-2 planetary computer

- [ ] model definitions `mef_agri.models`

  - [x] SWAT atmosphere
  - [ ] SWAT soil - TODO: check docstrings
  - [x] EPIC crop
  - [x] custom management
  - [x] INRAE Sentinel-2

- [ ] model evaluation

  - [x] statistical utilities for evaluation `mef_agri.evaluation.stats_utils.py`
  - [ ] database
  - [ ] model input
  - [ ] estimators (model propagation, bootstrap particle filter)

- [ ] utils
