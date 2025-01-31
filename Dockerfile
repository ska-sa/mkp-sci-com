# Use the official Ubuntu 22.04 LTS as a base image
FROM ubuntu:22.04

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y software-properties-common
RUN apt-add-repository -s ppa:kernsuite/kern-9 && \
    apt-add-repository multiverse && \
    apt-add-repository restricted && \
    apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    meqtrees \
    python3-meqtrees-cattery \
    libmeqtrees-timba0 \
    meqtrees-timba \
    python3-meqtrees-timba \
    kittens \
    python3-astro-kittens \
    purr \
    python3-purr \
    python3-astro-tigger \
    tigger \
    owlcat \
    python3-owlcat \
    python3-pyxis \
    pyxis \
    casacore-dev \
    casacore-tools \
    casarest \
    makems \
    qtcreator \ 
    qtbase5-dev \
    qt5-qmake \
    cmake \
    python3-pyqt5 \
    python-pyqt5.qwt-doc \
    python3-pyqt5.qwt \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-astlib \
    python3-casacore \
    python3-dev \
    build-essential \
    wget && \
    apt clean all && \
    apt autoremove

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV PATH=/usr/local/bin:$PATH
RUN ldconfig # update the linker config

ENTRYPOINT ["meqtree-pipeliner.py"]
CMD ["--help"]