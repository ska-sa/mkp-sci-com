fname = '1596017061_sdp_l0_split_10k+512_scans6.ms'
bscan = 1  # start scan to process
escan = 6 # end scan to process
win = 100 # window for detrending by polynomial filter
bchan = 10000 #  Beginning channel number of MS file if split (for flagging known RFI)
gbchan = 0 # bchan for gaincal
gechan = 64 # echan for gaincal
