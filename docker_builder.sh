#!/bin/bash
docker build -f Dockerfile . -t llms4ontomap

docker save -o llms4ontomap.tar llms4ontomap

singularity build llms4ontomap.sif docker-archive:llms4ontomap.tar
