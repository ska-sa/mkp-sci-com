# mkp-sci-com
MeerKAT+  Science Commissioning


This is the repository for the scripts and reduction configeration files for the MeerKAT+ Science Commissioning.


[Stimela documentation](https://stimela.readthedocs.io/en/latest/index.html)

[Stimela 2 paper](https://arxiv.org/abs/2412.10080)
# MeerKAT+ Sim with MeqTrees

**_This documentation is still a work in progress. Information will be added and removed as we learn more_**

## Objectives
Simulate MeerKAT+ heterogenous array with beams from holography measurements and produce some visibility to test calibration.

We will try to do this with MeqTrees.

## MeqTrees

### Overview
MeqTrees is a software package for implementing Measurement Equations, originally developed at ASTRON and now maintained by Rhodes University Centre for Radio Astronomy Techniques & Technologies (RATT).
It defines a Python-based Tree Definition Language (TDL) for building numerical expressions. 
MeqTrees implemented Smirnov's RIME (paper [I](http://www.aanda.org/articles/aa/full_html/2011/03/aa16082-10/aa16082-10.html), [II](http://www.aanda.org/articles/aa/full_html/2011/03/aa16434-11/aa16434-11.html), [III](http://www.aanda.org/articles/aa/full_html/2011/03/aa16435-11/aa16435-11.html), [IV](http://www.aanda.org/articles/aa/full_html/2011/07/aa16764-11/aa16764-11.html)) using TDL, which makes it suitable for radio interferometric simulation and calibration. 
A TDL script, defining the specific components of the RIME to compute, can be constructed and executed on the meqserver, the computational back-end of MeqTrees. 
A separate meqbrowser GUI is available for for parses TDL scripts and controlling meqservers although it is also possible to run meqservers in non-interactive "batch" mode

Several Python-based "framework" defining the specific building blocks and tasks (e.g. jones matrices, simulation, calibration, and etc.) to be computed have been defined in the [`meqtrees-cattery`](https://github.com/ratt-ru/meqtrees-cattery/tree/master) package. These framework are all cat-theme by the way. For example, the low-level framework is call Meow (Measurement Equation Object frameWork) while Siamese is the simultion framework based on Meow. [Oleg's presentation](https://raw.githubusercontent.com/wiki/ratt-ru/meqtrees/meqtrees-me.pdf) lists them all.

The actual underlying C++ code and software systems and (e.g. the meqserver) for MeqTrees is in the [`meqtress-timba`](https://github.com/ratt-ru/meqtrees-timba) repository. `meqtree-pipeliner.py`, the high-lelve entrypoint script for running MeqTrees in batch mode is also here.

Most pages in the [documentation Wiki](https://github.com/ratt-ru/meqtrees/wiki) has disappeared since ASTRON reworked its website, but the [MeqTrees paper](https://www.aanda.org/index.php?option=com_article&access=doi&doi=10.1051/0004-6361/201015013&Itemid=129), as well as the presentation linked above, still provide relevant information to understand the code.

### Installation

To get MeqTrees running, the easiest way is (probably) installing it on an Ubuntu machine via [Kernsuite](https://kernsuite.info/), which provide meqtrees, timba and cattery packages and (supposedly) everything else to run it.

Since we will be likely running it on a cluster, using a docker images and running MeqTree in the non-interative batch mode is probably what we need.

Kernsuite provides docker images that can be used as a based to build MeqTree docker image, but it is only availble for amd64 Linux, so we will build an Ubuntu-based docker image and add Kernsuite packages ourselves, so that it is compatible with arm64 (e.g. Apple silicon macOS) too.

The steps are:
1. Install Docker
2. Clone this repo
2. `cd` into the cloned repo and do `docker build --no-cache -t meqtree .`. This will build a docker image named meqtree. 

### Defining Heterogenous Array

This is somewhat documented in Ben Hugo's posts on this [GitHub issue](https://github.com/ratt-ru/meqtrees-cattery/pull/115)

### Beam Models
**_Still to dig into_**

It seems that several analytics models have been defined and numerical beams (e.g. holography or CST model) can be passed in as a beamfit file. 

Q: _Is there a standard for the beamfits format?_

## Progress Update

**2025-01-31:**
* Started these note and GitHub repo
* Built the docker image succesfully but ran into an issue related to the GUI and Qt when trying to run the entrypoint script non-interactively.
  * Contacted Ben Hugo for some advice
  * Will also try to run MeqTree interactively via Ubuntu virtual machine just to see what is going on.
* Several details w.r.t. how to define parameters in the TDL script needs to be looked into.

