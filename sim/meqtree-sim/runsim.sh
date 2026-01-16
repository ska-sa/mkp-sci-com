#!/bin/bash

start_freq=1300
channel_width=0.208984
n_channel=5
end_freq=$(echo "$start_freq + $n_channel * $channel_width" | bc)
echo $end_freq
ms=mkp77.6min.scan.8s.208khz.J0400-3300.ms
rm -r ${ms}

# Make empty measurement set with ssims
# Technically -T should be meerkat-plus, but this is link to casadata
# Might be better to just give the -lle
simms -dir "J2000,04h00m00.0s,-33d00m00s" -T meerkat\
    -dt 8 -st 0.1 -nc ${n_channel} -f0 ${start_freq}MHz -df ${channel_width}MHz \
    -pl XX YY -n ${ms} \
    -t ascii -cs itrf -l meerkat-plus \
    data/meerkat_plus.itrf.txt

# Make MeerKAT beam with eidos
eidos -p 256 -d 4 -f 1300 1350 5 \
    -P beam_models/mk.eidos.lband -o8

# Split holographic beam into meqtree format
python scripts/split_beam.py --prefix=mkp.holo.lband --outdir=beam_models data/MK+L_sym.fits

# Run stimela
stimela --backend singularity run cultcargo::meqtree-pipeliner \
    ms=${ms} skymodel=sky_models/single_point.txt \
    config=sim_config.tdl
    