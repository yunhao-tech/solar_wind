# RAMP starting kit on solar wind classification


_Authors of this challenge: Gautier Nguyen, Joris van den Bossche, Nicolas Aunai & Balazs Kegl_

#### Introduction

Interplanetary Coronal Mass Ejections (ICMEs) result from magnetic instabilities occurring in the Sun atmosphere, and interact with the planetary environment and may result in intense internal activity such as strong particle acceleration, so-called geomagnetic storms and geomagnetic induced currents. These effects have serious consequences regarding space and ground technologies and understanding them is part of the so-called space weather discipline.

ICMEs signatures as measured by in-situ spacecraft come as patterns in time series of the magnetic field, the particle density, bulk velocity, temperature etc. Although well visible by expert eyes, these patterns have quite variable characteristics which make naive automatization of their detection difficult.

The goal of this RAMP is to detect Interplanetary Coronal Mass Ejections (ICMEs) in the data measured by in-situ spacecraft.


#### Set up

install the dependencies via `environment.yml`

#### Local notebook

Get started on this RAMP with the [dedicated notebook](solar_wind_starting_kit.ipynb).

#### Your solution

Put your algorithm in `submissions/starting_kit/estimator.py`. The function `get_estimator()` should return a sklearn pipeline.

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.

---

# My roadmap for this data challenge

1. Feature engineering:
Adding some domaine specific features, c.f. the paper [Machine Learning Approach for Solar Wind
Categorization](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019EA000997): 
      - Plasma beta value
      - Dynamic pressure
      - Alfvén Mach number
      - Alfven speed
      - ram pressure
      - Ratio of proton and alpha number density
      - Fast magnetosonic Mach number
Also, I added the log scale for some features (for example, the Dynamic pressure) because of their small variance.
  
2. Compute the **rolling mean and rolling standard deviation on different time scale** (2h, 5h, 10h, 15h, 20h) for some features. Knowing that before and during the solar wind events, some parameters would have a huge fluctuation.

3. Post-processing: smooth the predicted probability.

4. Choose a good classifier: `HistGradientBoostingClassifier` in sklearn and tune its hyperparameters, especially `l2_regularization`, `min_samples_leaf` and `max_leaf_nodes`. 



