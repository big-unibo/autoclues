# AutoClues: Exploring Clustering Pipelines via AutoML and Diversification

This is the repository for the paper AutoClues: Exploring Clustering Pipelines via AutoML and Diversification* submitted to PAKDD 2024.
To implement the optimization process, we departed from the code provided in [2] (GitHub repository: [https://github.com/aquemy/DPSO_experiments](https://github.com/aquemy/DPSO_experiments)).

[1] A. Quemy, "Data Pipeline Selection and Optimization." DOLAP. 2019. http://ceur-ws.org/Vol-2324/Paper19-AQuemy.pdf

# Requirements

In order to reproduce the experiments in any operating systems, Docker is required: [https://www.docker.com/](https://www.docker.com/).
Install it, and be sure that it is running when trying to reproduce the experiments.

To test if Docker is installed correctly:

- open the terminal;
- run ```docker run hello-world```.

***Expected output:***

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete
Digest: sha256:7d246653d0511db2a6b2e0436cfd0e52ac8c066000264b3ce63331ac66dca625
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

# Reproducing the experiments

The instructions are valid for Unix-like systems (e.g., Linux Ubuntu, MacOS) and Windows (if using PowerShell).


Open the terminal and type:

```
docker run -it --volume ${PWD}/autoclues:/home/autoclues ghcr.io/anonymous-pakdd-24/autoclues:1.0.0
```

This creates and mounts the folder ```autoclues``` into the container (which is populated with the code and the necessary scenarios), and run the toy example.


# Customize the experiments

The structure of the project is the follow:

- ```.github``` contains the material for the GitHub action;
- ```datasets``` contains a dump of the leveraged datasets;
- ```experiment``` contains the source code;
- ```resources``` contains two options of search spaces;
- ```resources``` contains the results of the experiments;
- ```scenarios``` contains an example of the needed scenario;
- ```scripts``` contains the running scripts;
- ```.gitattributes``` and ```.gitignore``` are configuration git files;
- ```Dockerfile``` is the configuration file to build the Docker container;
- ```README.md``` describes the content of the project;
- ```requirements``` lists the required python packages.

Each individual experiment is described by a scenario written in YAML.

The folder ```scenarios``` should be filled with YAML files with format ```<dataset>_<metric>.yaml```.
The possible datasets are listed in the folder ```dataset```, the possible metrics are: ```sil``` (silhouette), ```ssw``` (sum of squares within clusters), and ```dbi``` (davies–bouldin index).

You can easily write your own scenario based on the following template:

```
general:
  dataset: syn0
  seed: 42
  space: extended
optimizations:
  smbo:
    budget: 7200
    budget_kind: time
    metric: sil-tsne
diversifications:
  mmr:
    criterion: clustering
    lambda: 0.5
    method: mmr
    metric: ami
    num_results: 3
runs:
- smbo_mmr

```


In particular, you can modify the following sections.
- In ```general```: you can modify field ```dataset``` by choosing among the ones in the folder ```dataset```.
- in ```optimizations```: you can modify the field ```budget``` by typing the time (in seconds) given to the optimzation, and the field ```metric``` by choosing among [```sil```, ```sil-tsne```, ```dbi```, ```dbi-tsne```]
- in ```diversifications```, you can modify the ```lambda``` factor.
