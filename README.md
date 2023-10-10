# Scientific Challenge and Direction Classification over Scientific Knowledge Graphs with Different Schemas

## Abstract

The output of research doubles at least every 20 years and in most research fields the number of research papers has become intractable.
It is possible to create a Knowledge Graph (KG), representing the content of the research body.
This KG can then be used to more efficient interact with this research content and also allows the use of Machine Learning models for graphs.
The KG is created according to a specific schema of choice. The present research investigates the effect of the schema on the ability to correctly classify whether a subgraph represents a future research suggestion (direction or challenge) or not.
Our results show that the SciERC schema yields the best performance across different settings, and that state of the art performance can be achieved when classifying graphs using typed relations and pre-trained embeddings. Overall, we observe that schemas with limited deviation in the resulting node degrees, and significant interconnectedness, lead to the best downstream graph classification performance.

## Contents

This repository contains the code used to generate the results for "Scientific Challenge and Direction Classification over Scientific Knowledge Graphs with Different Schemas".

Several scripts alot to verification, visualizations and tests. The pipeline takes the form of: "research sentences -> json files for dygie predictor -> dygie predictions -> full graphs -> full graph pretrained embeddings -> subgraphs -> subgraph".

| Script                                       | Data In                                                                  | Data Out                                                                                    | Process                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| run_pipeline.py                              | DanL/scientific-challenges-and-directions-dataset                        | KG_per_schema/data/predictions                                                              | raw research sentence docs -> research sentences -> json files for dygie predictor -> dygie predictions |
| get_full_graph_gcn.py                        | KG_per_schema/data/predictions                                           | full_schema_node_embeddings                                                                 | full graphs                                                                                             |
| build_full_graph_embeddings_relational.ipynb | full_schema_node_embeddings -> /content/drive/MyDrive/full_graphs/\*.pth | /content/drive/MyDrive/embedded/\*.pth -> embeddings_gdrive/                                | full graph pretrained embeddings                                                                        |
| gcn_data.py                                  | embeddings_gdrive/                                                       | gcn*subgraph_data_filtered/ -> /content/drive/MyDrive/subgraph_datasets/{schema}*{mode}.pkl | pretrained embeddings -> subgraphs                                                                      |
| subgraph_isresearch_classification.ipynb     | /content/drive/MyDrive/subgraph*datasets/{schema}*{mode}.pkl             | -                                                                                           | subgraphs -> is_research classification                                                                 |

## Usage

Ensure that you have python3.10 installed, as using python3.11 may cause package incompatibilities. For using the spacy schema's (not inclued by default) python3.8 is recommended.

```
git clone https://github.com/JoshuaSeth/VUThesis_LM_Triple_Extraction.git
cd VUThesis_LM_Triple_Extraction
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```
python KG_per_schema/run_pipeline.py
```
