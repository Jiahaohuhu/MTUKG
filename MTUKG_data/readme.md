### Guidance on data usage and processing

We store the raw, unprocessed files in the `./Meta_data` directory. To preprocess, align, and filter these files, we use the `preprocess_nyc_data.py` or `preprocess_chi_data.py` scripts. The processed data is then saved in the `./Processed_data` directory. The scripts `nyc_functional_zones.py` and `chi_functional_zones.py` are used to generate time-series functional zone data. Finally, we execute the `construct_TUKG_NYC.py` or `construct_TUKG_CHI.py` scripts to construct the urban knowledge graph; this process assigns unique IDs to the entities and relations within the MTUKG and partitions the graph into training, validation, and test sets, with the resulting graphs stored in the `./UrbanKG` directory.

File information for each directory is as follows:
```
./Meta_data Raw datasets: administrative division data, POI (Point of Interest) data, road network data, and urban spatiotemporal event data
./Processed_data Preprocessed data and functional zone clustering results
./MTUKG Different versions of MTUKG, featuring various entity types and diverse relationships
```

The following types of atomic files are defined:

| filename                    | content                                 | example                                 |
| --------------------------- | --------------------------------------- | --------------------------------------- |
| entity2id_XXX.csv           | entity_name, entity_id                  | area::110 11                            |
| relation2id_XXX.csv         | relation_name, relation_id              | FHPC 6                                  |
| static_train temporal_train | entity_id, relation_id, entity_id       | 1244001	26	721896	2022-06-28	2022-07-12 |
| static_vaild temporal_vaild | entity_id, relation_id, entity_id       | 1231969	24	22784	2016-03-16	2016-04-19  |
| static_test temporal_test   | entity_id, relation_id, entity_id       | 16379	2	98840	2017-07-01	2017-09-30     |
| UrbanKG_XXX.txt             | entity_name, relation_name, entity_name | road::8865	RLA	area::91                 |

