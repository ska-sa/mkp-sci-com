# MeerKAT+ Sim with MeqTrees

**_This documentation is still a work in progress. Information will be added and removed as we learn more_**

Simulate MeerKAT+ heterogenous array with beams from holography measurements and produce some visibility to test calibration using MeqTrees.

## MeqTrees

MeqTrees is a software package for implementing Measurement Equations, originally developed at ASTRON and now maintained by Rhodes University Centre for Radio Astronomy Techniques & Technologies (RATT).
It defines and uses a Python-based Tree Definition Language (TDL) for building numerical expressions. 

The Smirnov's RIME (paper [I](http://www.aanda.org/articles/aa/full_html/2011/03/aa16082-10/aa16082-10.html), [II](http://www.aanda.org/articles/aa/full_html/2011/03/aa16434-11/aa16434-11.html), [III](http://www.aanda.org/articles/aa/full_html/2011/03/aa16435-11/aa16435-11.html), [IV](http://www.aanda.org/articles/aa/full_html/2011/07/aa16764-11/aa16764-11.html)) is implemented in MeqTrees using the TDL, making it suitable for radio interferometric simulation and calibration. 

TDL scripts, defining the specific components of the RIME or tasks to compute, can be constructed and executed on the meqserver, the computational back-end of MeqTrees.
The `meqbrowser` GUI is available for parsing the TDL scripts and controlling `meqservers`, the actual underlying C++ code that perform the computation, although it is also possible to run meqservers in non-interactive "batch" mode via the `meqtree-pipeliner.py` CLI.

The [`meqtrees-cattery`](https://github.com/ratt-ru/meqtrees-cattery/tree/master) package is a collection of TDL scripts that serve as "framework" for the specific building blocks and tasks (e.g. jones matrices, simulation, calibration, and etc). These framework are all cat-theme by the way. For example, the low-level framework is call Meow (Measurement Equation Object frameWork) while Siamese is the simultion framework based on Meow. [Oleg's presentation](https://raw.githubusercontent.com/wiki/ratt-ru/meqtrees/meqtrees-me.pdf) lists them all. 

Most pages in the [documentation Wiki](https://github.com/ratt-ru/meqtrees/wiki) has disappeared since ASTRON reworked its website, but the [MeqTrees paper](https://www.aanda.org/index.php?option=com_article&access=doi&doi=10.1051/0004-6361/201015013&Itemid=129), as well as the presentation linked above, still provide relevant information to understand the code.

## Setup

To get MeqTrees running, the easiest way is installing it on an Ubuntu machine via [Kernsuite](https://kernsuite.info/), which provides meqtrees, timba and cattery packages and everything else to run it.

Since we will likely be running MeqTrees on a cluster, using a docker images and running it in the non-interative "batch" mode is likely the best approach.

Kernsuite provides docker images that can be used as a based to build MeqTree docker image, but it is only availble for amd64 Linux, so we will build an Ubuntu-based docker image and add Kernsuite packages ourselves, so that it is also compatible with arm64 platform (e.g. Apple silicon macOS) in case someone needs it.

To build the docker image:
1. Install Docker
2. Clone this repo
2. `cd` into the cloned repo and do `docker build --no-cache -t meqtree .`. This will build a docker image named meqtree. 

## Running `meqtree-pipeliner.py` script via the docker image.

The docker image is build with the `meqtree-pipeliner.py` as the entrypoint, meaning that `docker run meqtree` will execute this script in the container.

The particular TDL script in Siamese that we want to use is `turbo-sim.py`. This script is alrady inside the container (as we have built the container with the `meqtree-cattery` package). However, the path (in the container) to this script must be passed to the pipeliner script.

A TDL config file (usualy with `*.tdl` extension), containing simulation parameters and paths to sky models, beams and array configuration, must also be passed to the pipeliner script. The _work-in-progress_ config file, as well as sky models and other components are included in this repository.

Thus, to actually run the pipeliner script via the docker image, we need to also bind this repo on the host machine to a directory inside the container when running the docker image, so that the pipeliner can access the TDL config.

The actual command is:

```docker run -v $(pwd -P):/opt meqtree -c /opt/sim_config.tdl --mt=4 @turbo-sim /usr/lib/python3/dist-packages/Cattery/Siamese/turbo-sim.py =simulate```

This command will run bind the current directory (assume to be the cloned repo directory) to `/opt` in the container and run `meqtree-pipeliner.py` with `-c /opt/sim_config.tdl --mt=4 @turbo-sim /usr/lib/python3/dist-packages/Cattery/Siamese/turbo-sim.py =simulate` as arguments. Specifically, they do the following.
* `-c /opt/sim_config.tdl`:  path to the config file in the container, which we bind from the repo.
* `--mt=4`: run meqtree with 4 CPU threads
* `@turbo-sim`: parse the parameters in the `[turbo-sim]` section in the config file.
* `/usr/lib/python3/dist-packages/Cattery/Siamese/turbo-sim.py`: path to the TDL script in the container that we want to run.
* `=simulate`: run the simulate method in `turbo-sim.py`.

## Simulation Configuration and Components
**_IN PROGRESS_**

### Defining Heterogenous Array

This is somewhat documented in Ben Hugo's posts on this [GitHub issue](https://github.com/ratt-ru/meqtrees-cattery/pull/115), but we will need to experiment.

### Beam Models

It seems that several analytics models have been defined and numerical beams (e.g. holography or CST model) can be passed in as a beamfit file, but we will also need to experiment

### Sky Model

A cataloge of point source can be easiy defined. For example,

```
#format: ra_h ra_m ra_s dec_d dec_m dec_s i freq0 spi
04 13 26.400000 -81 0 0 1 1e6 0.0
```

Q: Can we define the source flux in multiple bands, and how does meqtree interpolate the flux over those bands?

### Data Container

MeqTree uses measurement set. An empty measurement set must be passed to the TDL script. How to make one matching simulation parameters?

Q: _Is there a standard for the beamfits format?_

## Progress Update

**2025-02-13:**
* Succesfully build a docker image to run MeqTree in batch mode via the pipeliner script.
* Consulting with Ben regarding the configuration of each of the simulation components.

**2025-01-31:**
* Started these note and GitHub repo
* Built the docker image succesfully but ran into an issue related to the GUI and Qt when trying to run the entrypoint script non-interactively.
  * Contacted Ben Hugo for some advice
  * Will also try to run MeqTree interactively via Ubuntu virtual machine just to see what is going on.
* Several details w.r.t. how to define parameters in the TDL script needs to be looked into.

## MeerKAT Holography Beam
https://archive-gw-1.kat.ac.za/public/repository/10.48479/wdb0-h061/index.html
This must be converted into the complex voltage beams format

## MeerKAT+ Holography Beam
This also must be converted.

## Sky Model
`simms` can only make MS with one field (one RA, Dec field center)
But sky models can have many sources around that fields