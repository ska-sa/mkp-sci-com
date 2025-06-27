#!/bin/bash

# Make empty measurement set with ssims
simms -dir "J2000,04h00m00.0s,-33d00m00s" -T meerkat \
    -dt 8 -st 0.5 -sl 0.16667 -nc 4096 -f0 856MHz -df 208.9843kHz \
    -pl XX YY -n mk64.0.5hr.10min.scan.8s.208khz.J0400-3300.m

# Make MeerKAT beam with eidos
eidos -d 4 -r 0.015625 -f 856 1712 5 -P mk.holo.lband.4deg -o8

# Run stimela
stimela --backend singularity run cultcargo::meqtree-pipeliner \
    -c tdls/ssmf_clean_sky.tdl --mt=4 @ssmf_clean $(TURBO_SIM) =simulate