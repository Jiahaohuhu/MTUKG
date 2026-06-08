## Installation
Create a python 3.8 environment and install dependencies:
```
conda create -n python3.8 MTUKG
conda activate MTUKG
pip install -r requirements.txt
```

## Directory Structure

- |-raw_data/: Store preprocessed atomic files.
- |-libcity/: Project code root directory.
    - |-config/: The ConfigParser class is defined here, which supports command line and config file to modify our default parameters.
    - |-data/: The Dataset class is stored in a subfolder of this folder according to different tasks.
    - |-model/: Model classes are stored in subfolders of this folder according to the tasks they belong to.
    - |-evaluator/: A task corresponds to a dedicated evaluator.
    - |-executor/: Each task provides a standard Executor, and the model can also have its own exclusive Executor.
    - |-pipeline/: Store user-oriented pipeline functions, which are responsible for running through the entire framework process.
    - |cache/: Store the cache. Specifically, data preprocessing results, model training results, and evaluation results will be cached.
    - |-tmp/: Store temporary files such as checkpoint generated during training.
    - |-utils/: Store some general utility functions.
- |-log/: Store log information during training.

The directory structure could help readers better understand our code framework.
## Datasets
The complete static data can be found in the [Folder - Google Drive](https://drive.google.com/drive/folders/1egTmnKRzTQuyW_hsbFURUonGC-bJmBHW), and the complete time-series data can be found at XXX.

The following types of atomic files are defined:

|filename|content|example|
|---|---|---|
|xxx.geo|Store geographic entity attribute information.|geo_id, type, coordinates|
|xxx.rel|Store the relationship information between entities, such as areas.|rel_id, type, origin_id, destination_id|
|xxx.dyna|Store traffic condition information.|dyna_id, type, time, entity_id, location_id|
|config.json|Used to supplement the description of the above table information.||

we explain the above four atomic files as follows:

**xxx.geo**: An element in the Geo table consists of the following four parts:

**geo_id, type, coordinates.**

```
geo_id: The primary key uniquely determines a geo entity.
type: The type of geo. These three values are consistent with the points, lines and planes in Geojson.
coordinates: Array or nested array composed of float type. Describe the location information of the geo entity, using the coordinates format of Geojson.
```

**xxx.rel**: An element in the Rel table consists of the following four parts:

**rel_id, type, origin_id, destination_id.**

```
rel_id: The primary key uniquely determines the relationship between entities.
type: The type of rel. Range in [usr, geo], which indicates whether the relationship is based on geo or usr.
origin_id: The ID of the origin of the relationship, which is either in the Geo table or in the Usr table.
destination_id: The ID of the destination of the relationship, which is one of the Geo table or the Usr table.
```

**xxx.dyna**: An element in the Dyna table consists of the following five parts:

**dyna_id, type, time, entity_id(multiple columns**.

```
dyna_id: The primary key uniquely determines a record in the Dyna table.
type: The type of dyna. There are two values: label (for event-based task) and state (for traffic state prediction task).
time: Time information, using the date and time combination notation in ISO-8601 standard, such as: 2020-12-07T02:59:46Z.
entity_id: Describe which entity the record is based on, which is the ID of geo or usr.
```

**xxx.config**: The config file is used to supplement the information describing the above five tables themselves. It is stored in `json` format and consists of six keys: `geo`, `usr`, `rel`, `dyna`, `ext`, and `info`.

## Quick to Usage
The script used for training and evaluating a single model is provided in the root directory of the framework, and a series of command line parameters are provided to allow users to adjust the running parameter configuration.`run_model.py`

When run the , you must specify the following three parameters, namely , and . For example:`run_model.py``task``dataset``model`

```shell
python run.py --task traffic_state_pred --model STGCN --dataset NYCTaxi20200406
```

This script will run the STGCN model on the NYCTaxi20200406 dataset for traffic state prediction task under the default configuration.

**How to fuse UrbanKG embedding?**
To integrate UrbanKG embeddings, we directly concatenate the embeddings with USTP features to form the input. You can make modifications in **`./kg_graph_nyc.json`** and **`./kg_graph_chi.json`**.

## References

Some of the code was forked from the original LibCity [1] implementation which can be found at: [https://github.com/LibCity/Bigscity-LibCity-Docs-zh_CN](https://github.com/LibCity/Bigscity-LibCity-Docs-zh_CN)

[1] Wang, Jingyuan, et al. "Libcity: An open library for traffic prediction." _Proceedings of the 29th international conference on advances in geographic information systems_. 2021.