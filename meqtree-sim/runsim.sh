#!/bin/bash

freq=1391

# Make empty measurement set with ssims
# Technically -T should be meerkat-plus, but this is link to casadata
# Might be better to just give the -lle
simms -dir "J2000,04h00m00.0s,-33d00m00s" -T meerkat\
    -t ascii -cs itrf -l meerkat-plus \
    -dt 8 -st 0.1 -nc 1 -f0 1391MHz -df 208.984kHz \
    -pl XX YY -n mkp77.6min.scan.8s.208khz.J0400-3300.ms \
    meerkat-plus.itrf.txt

# Make MeerKAT beam with eidos
eidos -p 256 -d 4 -f 1400 14000.3 0.208984 -P beam_models/mk.holo.lband.1391MHz.4deg -o8

# Run stimela
stimela --backend singularity run cultcargo::meqtree-pipeliner \
    --ms mkp77.6min.scan.8s.208khz.J0400-3300.ms --skymodel sky_model.txt \
    --config sim_config.tdl
    