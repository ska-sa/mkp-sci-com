# Use the official Ubuntu 22.04 LTS as a base image
FROM ubuntu:22.04

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-add-repository -s ppa:kernsuite/kern-9 && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    meqtrees-timba \
    python3-meqtrees-cattery \
    python3-astro-tigger-lsm \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-astlib \
    python3-casacore \
    build-essential \
    wget && \
    apt clean all && \
    apt autoremove
RUN pip install simms eidos jupyterlab

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH=/usr/local/bin:$PATH
RUN ldconfig # update the linker config

ENTRYPOINT ["meqtree-pipeliner.py"]
CMD ["--help"]