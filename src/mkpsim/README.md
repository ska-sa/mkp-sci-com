# mkpsim

Stimela workflow recipe for simulating the MeerKAT+ using meqtrees.

## Installation

Clone the repository, and run `pip install .` from the the project [root](../../). This will install the `mkpsim` package and neccesary modules.

## mkpsim simulation recipe

The mkpsim package comes with a stimela recipe of the same name, which chain several tools into steps that ultimately produce a simulated MS file and quick wsclean dirty fits images.

See the [cabs definition](custom-cabs.yml) for the detail of each step. Notably, the recipe uses the MeerKAT beam model from [ `eidos` ](https://github.com/ratt-ru/eidos) package. The quick-image step is currently hard-corder to primarily produce dirty image ( `wsclean niter=1 ...` ).

```bash
$ stimela doc mkpsim::mkpsim 
...
└── Recipe: mkpsim
    ├── Description:
    │   └── Stimela recipe for MeerKAT+ simulation                                                                                                                           
    ├── Required inputs:
    │   └── outdir         Directory  Output directory. Will be created if not exist.
    │       skymodel       File       Source catalog file.                           
    │       mkp-beam-file  File       MeerKAT+ beam FITS file.                       
    │       antpos         File       Antenna position file.                         
    │       sim-config     File       meqtree configuration file.                    
    ├── Optional inputs:
    │   └── ncpus                int    Number of CPUs to use for meqtree-pipeliner. [default: 8]                                 
    │       array                str    Array type, either "hetero" or "homo" [default: hetero]                                   
    │       prefix               str    Prefix of the output files. [default: mkp.sim]                                            
    │       direction            str    Pointing direction. Example J2000,0h0m0s,-30d0m0s. [default: J2000,04h00m00.0s,-33d00m00s]
    │       freq-start           float  Start frequency in MHz. [default: 1300]                                                   
    │       freq-end             float  End frequency in MHz. [default: 1305]                                                     
    │       channel-width        float  Channel width in MHz. [default: 0.208984]                                                 
    │       duration             float  Observation duration hours. Default is 0.1. [default: 0.1]                                
    │       integration          float  Integration time in seconds. Default is 4. [default: 4.0]                                 
    │       imsize               int    Size of the quick image. [default: 1024]                                                  
    │       imscale              str    Resolution of the quick image. [default: 0.01deg]                                         
    │       primary-beam-enable  bool   Apply primary beam to sky model. [default: True]                                          
    ├── Obscure inputs: omitting 169
    ├── Obscure outputs: omitting 2
    └── Steps:
        └── mkdir                 Create output directory if it does not exist.                  
            make-meerkat-beam     Make MeerKAT beam models with eidos                            
            split-mkp-beam        Split MeerKAT+ holographic beam into 8 complex voltage beams   
            create-empty-ms       Make an empty measurement set with simms                       
            generate-stationspec  Generate stationspec JSON file                                 
            simulate-vis          Simulate visibility with meqtree-pipeliner                     
            quick-image           Make quick image with wsclean   
```

Among the 5 required inputs, 
* `mkp-beam-file` is the [MeerKAT+ Holography Beam models](https://drive.google.com/drive/folders/1otR21f4uGo2WUBH5HmudoNO142TRpnpX?usp=sharing). These must be mnaully dowloaded.
* `skymodel` should be an ASCII file point source catalog in [Tigger format](https://github.com/ratt-ru/tigger-lsm/blob/ab868c7db8423d7153d0c2eb48c7ca29fd84d287/Tigger/Models/Formats/ASCII.py#L45-L87). Some examples are provided in the [mkpsim package source code](./sky_models/)
* `antpos` is MeerKAT+ antenna position file. Also, provided in the [mkpsim package source code](./configs/meerkat_plus.itrf.txt)
* `sim-config` is additional configuration parameters for meqtree-pipeliner script not defined in this recipe. A simple config is provided in the [mkpsim package source code](./configs/meqtree_config.tdl).

## Running the simulation

To run the simulation, do, for example

```
stimela run mkpsim::mkpsim ncpus=24 outdir=sim_results/grid_10x10_2deg_hetero skymodel=sky_models/grid_10x10_2deg.txt mkp-beam-file=beam_models/MK+L_sym.fits antpos=src/mkpsim/configs/meerkat_plus.itrf.txt sim-config=src/mkpsim/configs/meqtree_config.tdl array=hetero prefix=mkp.1h.4s.5MHz.00h00m-30d00m direction='J2000,00h00m00s,-30d00m00s' freq-start=1300 freq-end=1305 duration=1.0 integration=4.0 imsize=4096 imscale=2asec
```

## Outputs

Upon succesful simulation, the specified output will consists of:
* `beams` directory containining the 8-polarization voltage beamfits file extracted from the holography beam.
* `logs` directory containing stimela logs for each run and each step.
* an output MS file
* a `stationspec*.json` file. The `generate-stationspec` creates this file by modifying the template in the [subpackage config](./configs/) to pass the correct voltage beamfits paths to meqtree
* fits images output from wsclean `quick-image` step.

## mkpsim.utils

The `utils` module of the `mkpsim` package provides several utility functions, some of which are used in the steps in the stimela recipe. Among these, `generate_point_soure_grid` can be used to generate a grid of point sources, `split_beam` can be use to split a beamfits beam into a set of 8-pol voltage beams, and `plot_image` can be used to visualise the output fits images, including plotting a difference between two images.
