# Literature Review

## Why knowledge graphs?
Wilcke, Xander, Bloem, Peter, and de Boer, Victor. ‘The Knowledge Graph as the Default Data Model for Learning on Heterogeneous Knowledge’. 1 Jan. 2017 : 39 – 57. https://content.iospress.com/articles/data-science/ds007

Wilcke and al. make the case that knowledge graphs should be the default model for machine learning on heterogeneous knowldege. They compare knowledge graphs to other approaches: tabular data, XML and relational databases. Their conclusion is that knowledge graphs are a better solution when knowledge is heterogeneous (different types of relationships exist in the data), incomplete (either edges or vertices are missing) and non-strictly hierarchical (for example a social network). Knowledge graphs can also be simply extended by concatenation between two or more knowledge graphs, provided that the knowledge encoding scheme isn't too dissimilar. 


## Health Knowledge Graph

Rotmensch, M., Halpern, Y., Tlimat, A. et al. Learning a Health Knowledge Graph from Electronic Medical Records. Sci Rep 7, 5994 (2017). https://doi.org/10.1038/s41598-017-05778-z

Rotmensh and al. constructed a disease-symptoms knowledge graph from patient record data. They used string matching to identify concepts (both disease and symptoms) in the patient records, and statistical models to learn the edges between concepts. The three models used where logistic regression, naive Bayes and noisy OR gates Bayesian network. The noisy OR gates performed better when evaluated against the Google Health Knowledge Graph. The approach demonstrates the feasability of unsupervised relation extraction for health data, as long as concepts can be extracted by string matching, which implies that concepts are preexisting and that the researchers have priors about which concepts are likely to be found in the data and relevant.

## OpenNRE framework

@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}

Han, Xu and Gao have introduced OpenNRE, an opensource and extensible toolkit to implement relation extraction models. The framework facilitates experiments with neural relation extraction methods.

## Using side information with RESIDE

https://paperswithcode.com/paper/reside-improving-distantly-supervised-neural

Distant-supervised RE make use of an exsiting Knowledge Base to automatically extract similar relations from unstructured text. Vashishth and al. show that additional side information from the KB can improve relation extraction. They use Graph Convolution Networks to encode syntactic information from text. Their method achieves state of the art results on the New York Times Corpus.    