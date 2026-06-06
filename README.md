

# MTUKG: A Multi-scale Temporal Urban Knowledge Graph Dataset for Knowledge-Enhanced Spatiotemporal Prediction

## 1. Overview

![FZ](FZ.png)

MTUKG is a multi-scale temporal urban knowledge graph dataset designed for knowledge-enhanced urban spatiotemporal prediction. It organizes heterogeneous urban entities across micro-, meso-, and macro-levels, including POIs, roads, junctions, functional zones, areas, and boroughs. Unlike conventional static UrbanKGs, MTUKG represents urban evolution through time-interval facts, enabling urban knowledge to be aligned with specific prediction windows.

The dataset introduces temporal functional zones constructed from spatial units by jointly considering static urban contexts, spatial morphology, road structures, POI spatial influence, and dynamic urban events. By modeling both static spatial topology and temporal urban changes, MTUKG supports multi-scale temporal reasoning and provides structured knowledge for downstream tasks such as taxi demand prediction, crime prediction, 311 service request prediction, and POI-level knowledge graph completion.

This repository provides the dataset, construction scripts, knowledge graph embedding modules, and evaluation code for reproducing the experiments of MTUKG.

## 2. Installation
You can create and activate the environment required to run the project using the following commands.

```
conda create -n python3.8 MTUKG
conda activate MTUKG
pip install -r requirements.txt
```

Please ensure that you have cloned the project and entered the directory before running the above commands.



