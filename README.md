Repository containing several disjointed tasks and investigations for the thesis at VU: Triple extraction from future research sentences


**Comparing different NER + RE extractors**
Comparison of different datasets and schema's by running the streamlit_app.py with streamlit run.

The ORKG_parsers folder has several parsers that do joint NER and RE. For example Dygie, see: https://github.com/dwadden/dygiepp#making-predictions-on-existing-datasets.

Datasets need to be preprocessed to be used with dygie, see: https://github.com/dwadden/dygiepp/blob/master/doc/data.md#formatting-a-new-dataset

A spacy interface is also present in the dygiepp repo, on which a thesis interface is added, but currently this only works when using the SciErc-trained dygie model.

