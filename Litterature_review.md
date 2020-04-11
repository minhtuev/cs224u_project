# Authors:
- Julien Laurent
- Vijay Ramakrishnan Thimmaiyah
- Vo Thanh Minh Tue

# Problem Definition:

Drug-drug interactions (DDI) is a phenomenon when one drug influences the level or activity of another drug. Extraction of DDis is significantly important for public health safety since it’s reported that about 2 million people in the USA aged 57 to 85 took potentially dangerous combinations of drugs (Landu, 2009) and Payne, 2007 mentioned that the deaths from accidental drug interactions rose by 68% from 1999 to 2004.

The most instructive dataset for DDI is the DDIExtraction 2013 dataset that contains 1,017 sentences that are manually annotated with a total of 18,491 pharmacological substances and 5,021 drug-drug interactions. The DDIExtraction 2013 dataset was part of SemEval2-13 task 9 for the extraction of drug-drug interactions from biomedical texts. The team (FBK-irst) with the best model scored 65.1% F-1 score on the DDIExtraction 2013 task-2 dataset. 

Our team wants to use the DDIExtraction 2013 dataset as a knowledge base that is augmented through distant supervision by sentences from pharmacological research papers to perform relation extraction for DDI without explicit labelled data. With the augmented dataset, we intend to use a transformer architecture to extract DDIs. We think that the distantly supervised augmented dataset and transformer architecture will provide state-of-the-art results on DDIExtraction 2013. We also want to evaluate the model on the open Covid-19 dataset and see if we can find novel DDI.


# Concise summary of articles:

We present a summary of articles on DDI extraction based on a historical timeline. We also present the articles based on general techniques of representing healthcare data and then diving into the specifics of the DDI extraction problem.

## Why knowledge graphs?
Wilcke and al. (2017) make the case that knowledge graphs should be the default model for machine learning on heterogeneous knowledge. They compare knowledge graphs to other approaches: tabular data, XML and relational databases. Their conclusion is that knowledge graphs are a better solution when knowledge is heterogeneous (different types of relationships exist in the data), incomplete (either edges or vertices are missing) and non-strictly hierarchical (for example a social network). Knowledge graphs can also be simply extended by concatenation between two or more knowledge graphs, provided that the knowledge encoding scheme isn't too dissimilar.

## Health Knowledge Graph – Manual Era
Thanks to the modular nature of knowledge graphs, they have been used in medicine for a long time. One of the first such systems was MYCIN (Van Melle, 1978). It was a consultation system designed to help physicians with the diagnosis and treatment of bacterial infections. The knowledge graph was composed of 350 production rules that were encoded by infectious disease experts.
In medicine, the early focus was on inference. INTERNIST-1/QMR (Swe et al, 1991) was an expert system that could provide clinicians with diagnostic cases. QMR was leveraging a belief-network representation of the knowledge base, which could be used for inference. That is, given observed symptoms – encoded as a binary variable – the system would compute the probability of a disease, and provide a decision-theoretic choice of diagnosis. Effective inference necessitated some simplifying assumptions, namely marginal independence of disease, conditional independence of findings, binary disease and findings, as well as causal independence of diseases. Overall, it was performing well: the system performed comparably to hospital clinicians on cases from clinic-pathological conferences and even performed better than a ward team on diagnosing challenging cases in one study. It’s main weakness, however, wasn’t inference. The knowledge base itself took over 20 person-years of work to develop. It contained over 600 diseases and 4,000 findings (i.e. symptoms). The time and cost required is tremendous, which motivated the research for automatic extractions of relations in unstructured text data.

## Health Knowledge Graph – Automated Era 
In more recent years, researchers have attempted to use machine learning algorithms to extract relations and build knowledge graphs from unstructured text data. Rotmensh and al. (2017) constructed a disease-symptoms knowledge graph from patient record data. They used string matching to identify concepts (both disease and symptoms) in the patient records, and statistical models to learn the edges between concepts. The three models used were logistic regression, naive Bayes and noisy OR gates Bayesian network. The noisy OR gates performed better when evaluated against the Google Health Knowledge Graph. The approach demonstrates the feasibility of relation extraction for health data, as long as concepts can be extracted by string matching, which implies that concepts are preexisting and that the researchers have priors about which concepts are both likely to be found in the data and relevant. As a further limitation, the relations extracted are all of one type only: “disease A causes symptom B”.

In order to transcend these limitations, Ernst P. et al (2015) used distant supervision techniques to create the KnowLife knowledge graph. They take as a source a small knowledge base of 467 seed facts, 13 binary relations, the Unified Medical Language System dictionary of biomedical entities, and a large text corpus consisting of scientific articles abstracts, full text as well as other text data from Web portals and online communities. The processing pipeline is composed of an entity recognition stage, a pattern gathering stage, a pattern analysis stage and finally a consistency reasoning stage. The authors use the system to generate the KnowLife KB, consisting of 542,689 facts for 13 different relations. The average precision is 93%, as determined sampling and manual assessment. While that system’s performance is impressive, the pipeline is very complex, and heavily customized for the task at hand. Many of the design choices seem arbitrary. For example, the entity recognition stages make use of string similarity matching on a character-level 3-grams. The paper doesn’t explain why word similarity, maybe with subword enrichment, wouldn’t work as well. Similarly, the use of Jacquard distance for pattern analysis instead of the more commonly used cosine distance is not justified. To be fair to the authors, the space of possible design choices was so large that we can’t really expect them to explore it all or to justify every sub-step. Such a complex and customized pipeline however puts into question the replicability of the work: would a different set of seed facts, or different binary relations, require a completely different system? In particular interest for us, it’s not clear that the Ernst et al. approach would work equally well for the Drug-Drug interaction problem.

## Health Knowledge Graph – Drug-Drug interaction 
One of the most influential datasets in knowledge graphs for DDI is the SemEval-2013 Task 9 dataset (Segura-Bedmar et al, 2013). The dataset contains 1,107 texts from DrugBank (784) and MedLine (233). The task had two problems: drug entity detection in sentences (task 1) and DDI (task 2). We are interested in task 2 since our research focus is on DDI. The format of the data are XML documents containing a sentence, all the extracted drug names and their interactions. One interesting observation of the paper is that the data is very skewed: there are a lot more negative interaction dataset.

8 teams participated in the task workshop. Most of the participating systems were built on support vector machines (SVMs). Approaches based on non-linear kernel methods achieved better results than linear SVMs. Most systems primarily used syntactic information but semantic information was poorly understood. The best system was a non-linear SVM kernel that got a F-1 score of 65.1%.

We plan to use SemEval-2013 dataset as the gold knowledge-base for DDI that we want to augment using distance supervision. However, instead of 5 possible labels of interaction, we reduce this to 2: interaction or no-interaction.

## Traditional ML approaches

### Support Vector Machines (SVMs)
The best paper on SemEval-2013 was a non-linear Kernel SVM from Chowdhury et al, 2013. The paper first segmented the training data into relevant and non-relevant examples. The non-relevant filter used two techniques. The first technique exploited the scope of negations in sentences to remove drugs that are not related to each other, for example, “drug A does not interact with drug B”. Such types of sentences were common in the SemEval-2013 dataset. The second technique used some filters like “if only one entity is present, ignore it” or “if one entity and it’s synonym are present, ignore the relation”. Once the non-relevant data was sieved out, the model then used a hybrid non-linear kernel SVM model for DDI. This paper had an F1 score of 65.1% F-1 score on the DDIExtraction 2013.

Bokaraeian et al, 2013 used a combination of word features including bag-of-words, part of speech tags, constituency parse tree features, conjunction, verbs and negation features to create a feature set that was then passed to a classifier. The paper tried different classifiers including random forest, Naive Bayes and SVMs and found that the SVM was the best model on the validation set. However, their F1 score was lower than Chowdhury et al, 2013 at 53.5%.

### Ensemble approaches
Philippe, et al, 2011 used an ensemble model based on three models: Kernel classifiers and a case-based reasoning model ensembled in a majority-voting policy to classify the drug interaction. The ensemble approach got a F-1 score of 60.6%. 

## Deep Learning approaches

Deep learning for drug–drug interaction extraction from the literature: a review
This is a comprehensive survey paper by Zhang et al that attempts to summarize different papers and approaches for using Deep Learning (DL) for DDI extraction, given its ascendency in natural language processing, speech recognition and computer vision. The major DL frameworks for DDI are Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Recursive Neural Network (Recursive NN). These approaches are detailed and contrasted in the next section.

Here we include two other papers that are interesting to us but not included in the Zhang et al survey.
BERE: An accurate distantly supervised biomedical entity relation extraction network
Hong et al (2019) used distant supervision and a deep learning model that used Gumbel Tree Gated Recurrent Unit to learn sentence embeddings that incorporated entity information. They also used word-level attention for improving DDI relation extraction and sentence-level attention to gather distant supervised data. The distant supervision dataset was all of PubMed aligned with drug entities from drugbank. Their F1 scores on DDIExtraction 2013 was 73.9%, significantly higher than the traditional ML model accuracies.

Chemical–protein interaction extraction via contextualized word representations and multihead attention
In this work, Yijia Zhang et al  proposed to integrate the deep context representation, Bi-LSTMs and multihead attention in Chemical-Protein Interaction (CPI) extraction and evaluated on the recent ChemProt corpus. The evaluations show that both deep context representation and multihead attention approach do improve the performance in CPI extraction and achieved the highest performance of 0.659 in F-score on the ChemProt corpus. The authors also experimented with DDI 2013, therefore suggesting that this approach can be generalizable to other biochemical tasks.





# Compare and Contrast:

Traditional ML models applied on DDIExtraction like Chowdhury et al, 2013 and Bokaraeian et al, 2013 seem to perform quite poorly on the evaluation set. It’s noted that SVMs with non-linear kernels outperformed all the other models. Moreover, Chowdhury et al, 2013 outperforms all the other traditional models like Bokaraeian et al, 2013 and Philippe, et al, 2011. It is interesting to note that Philippe, et al, 2011 does implement SVMs as one of their ensemble models but don’t outperform Chowdhury et al, 2013. This might be due to Chowdhury et al, 2013 have very good non-interaction filters to filter out sentences that don’t have any interactions in them. Chowdhury et al, 2013 also has done a lot of prior art on building these non-relevant classifiers, including a negate classifier discussed in the paper. It might be useful to use these non-interaction heuristic models for the distant supervision we propose to use.  

The traditional ML models like SVMs rely heavily on tedious feature engineering and redundant feature selection moreover, a supervised definition of the feature set would reduce the chance to discover other valuable patterns. Therefore, deep learning models would show more promise in this domain as they have done in other relation extraction domains.

Based on the framework given by Zhang et al we can compare and contrast different DL approaches. Among different CNNs, there are conventional CNNs (Liu et al and Quan et al), dependency-based CNNs (Zhao et al), deep CNNs (Dewi et al, Sun et al), molecular-structure/graph based CNNs (Masaki et al), and attention-based CNNs (Masaki et al). 

Among different RNNs, there are conventional RNNs (Ramakanth et al, Huang et al, Jiang et al), dependency-based RNNs (Wang et al), and attention-based RNNs (Zheng et al, Yi et al, Zhou et al, Sunil et al, Xu et al).

Some researchers attempt to use recursive RNNs to capture semantic compositionality. Victor et al uses a recursive NN to detect and classify DDIs from biomedical texts based on Matrix-Vector spaces, but the result was not outstanding. Sangrak et al designed a new recursive NN model, the tree LSTM, to achieve good results in experiment.

Among different approaches, CNNs were the earliest models, which achieved some good results but could not handle long sentences. Deeper CNNs can address some of these issues but require more time to train and higher computational costs. RNNs can address arbitrary input length, and dependency trees can help CNNs and RNNs to pay close attention to words around important parts in a parser tree as well as relation between two entities.

The attention-based methods do provide more targeted semantic matching, which can help models find important words and recognize contextual information explicitly. Applying attention mechanism has improved performances for both short and long sentences. Among all models, according to the survey by Zhang et al, Attention-based BiLSTM (Zheng et al) achieves state of the art accuracy on the detection problem; Deep CNN (Dewi et al) achieves state of the art accuracy on negative instance filtering; Deep CNN2 (Sun et al) achieves state of the art accuracy on no negative instance filtering, followed by Attention-based BiLSTM.

All supervised DL approaches suffer from common problems: lack of training data, instability of the model performance and interpretability. Among these issues, the lack of training data has proven to be a persistent bottleneck especially problematic in biomedical fields where data are mostly unstructured. Currently all models are trained and tested with the DDI corpus. To address this problem, semi-supervised learning and distant supervision are most promising. Semi-supervised learning is already used in most of the DDI models. Distant supervision is another promising approach since it can leverage extensive chemical databases, but it has two weaknesses: wrong labeling and sparsity. Wrong labeling can be addressed with avoiding noisy labels and recognizing wrong labels; sentence analysis and knowledge graph might be useful in this regard. Inspired by GANs, Qin et al propose a generative adversarial training for distant supervision; in general, future works in Reinforcement Learning (RL) and GANs can be useful to address this problem. Regarding data sparsity, Zhang et al recommend integrating distributional module and pattern module. 

Regarding performance instability, Zhang et al suggest joint learning, ensemble learning and dual learning.

# Future work
It’s clear from the literature survey that deep learning models would be able to address DDI effectively. However, the current state-of-the-art deep learning approaches do not leverage attention networks using Transformers, which have shown promising results in NLU tasks. Thus, there is a need to explore transformer network architectures like BERT for DDI.

Moreover, the current literature for DDI does not use the latest scientific dataset relevant to academia right now. There is no mention of using the CORD-19 dataset for mining DDI between Covid-19 drugs and well-known drugs used to treat other illnesses. We plan to evaluate our DDI model on CORD-19 and validate the mined interactions against the gold-label knowledge base of known Covid-19 DDI (Interactions with Experimental COVID-19 Therapies, Liverpool Drug Interactions Group) which are currently not present in the SemEval 2013 dataset. 




# Bibliography
- Asada, Masaki, Makoto Miwa, and Yutaka Sasaki. "Enhancing drug-drug interaction extraction from texts by molecular structure information." arXiv preprint arXiv:1805.05593 (2018).
- Asada, Masaki, Makoto Miwa, and Yutaka Sasaki. "Extracting drug-drug interactions with attention CNNs." BioNLP 2017. 2017.
- Bokharaeian, Behrouz, and Alberto Díaz. "NIL_UCM: Extracting Drug-Drug interactions from text through combination of sequence and tree kernels." Second Joint Conference on Lexical and Computational Semantics (* SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013). 2013.
- Chowdhury, Md Faisal Mahbub, and Alberto Lavelli. "FBK-irst: A multi-phase kernel based approach for drug-drug interaction detection and classification that exploits linguistic information." Second Joint Conference on Lexical and Computational Semantics (* SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013). 2013.
COVID-19 Open Research Dataset (CORD-19). 2020. Version 2020-03-20. Retrieved from https://pages.semanticscholar.org/coronavirus-research. Accessed 2020-04-06. doi:10.5281/zenodo.3715505
Dewi IN, Dong S, Hu J. Drug–drug interaction relation extraction with deep convolutional neural networks. 2017. IEEE International Conference on Bioinformatics and Biomedicine (BIBM). 2017. IEEE Comp Soc.
- Ernst, P., Siu, A. & Weikum, G. KnowLife: a versatile approach for constructing a large knowledge graph for biomedical sciences. BMC Bioinformatics 16, 157 (2015). https://doi.org/10.1186/s12859-015-0549-57
- Hong, Lixiang, et al. "BERE: An accurate distantly supervised biomedical entity relation extraction network." arXiv preprint arXiv:1906.06916 (2019).
Interactions with Experimental COVID-19 Therapies (www.covid19-druginteractions.org) Accessed 2020-04-06.
- Jiang Z, Gu L, Jiang Q. Drug drug interaction extraction from literature using a skeleton long short term memory neural network. 2017 IEEE Int Conf Bioinform Biomed. 2017. IEEE.
- Kavuluru, Ramakanth, Anthony Rios, and Tung Tran. "Extracting drug-drug interactions with word and character-level recurrent neural networks." 2017 IEEE International Conference on Healthcare Informatics (ICHI). IEEE, 2017.
- Qin P, Xu W, Dsgan WWY. Generative adversarial training for distant supervision relation extraction. Proceedings of the 56th Annual Meeting of the. Association for Computational Linguistics. 2018;496–505.
- Quan, Chanqin, et al. "Multichannel convolutional neural network for biological relation extraction." BioMed research international 2016 (2016).
- Rotmensch, M., Halpern, Y., Tlimat, A. et al. Learning a Health Knowledge Graph from Electronic Medical Records. Sci Rep 7, 5994 (2017). https://doi.org/10.1038/s41598-017-05778-z  
- Sahu, Sunil Kumar, and Ashish Anand. "Drug-drug interaction extraction from biomedical texts using long short-term memory network." Journal of biomedical informatics 86 (2018): 15-24
- Sangrak L, Kyubum L, Jaewoo K, et al. Drug drug interaction extraction from the literature using a recursive neural network. PLoS One 2018;13(1).
- Segura Bedmar, Isabel, Paloma Martínez, and María Herrero Zazo. "Semeval-2013 task 9: Extraction of drug-drug interactions from biomedical texts (ddiextraction 2013)." Association for Computational Linguistics, 2013.
- Shengyu L, Buzhou T,Qingcai C, et al. “Drug–drug interaction extraction via convolutional neural networks.” Computational Methods Med 2016;2016:1–8.
- Shwe, M. A. et al. Probabilistic diagnosis using a reformulation of the INTERNIST-1/QMR knowledge base. Methods of information in medicine 30, 241–255 (1991).
- Suárez-Paniagua, Víctor, and Isabel Segura-Bedmar. "Using recursive neural networks to detect and classify drug-drug interactions from biomedical texts." Proceedings of the Twenty-second European Conference on Artificial Intelligence. IOS Press, 2016.
- Sun X, Feng J,Ma L, et al. Deep convolution neural networks for drug–drug interaction extraction. IEEE Int Conf Bioinform Biomed. IEEE, 2018, 2018.
- Thomas, Philippe, et al. "Relation extraction for drug-drug interactions using ensemble learning." Training 4.2,402 (2011): 21-425.
- Tianlin, Zhang, et al. “Deep learning for drug–drug interaction extraction
Van Melle, W. MYCIN: a knowledge-based consultation program for infectious disease diagnosis. International Journal of Man-Machine Studies 10, 313–322 (1978).
- Wang W, Yang X, Yang C, et al. Dependency-based long short term memory network for drug–drug interaction extraction. BMC Bioinform 2017;18(16):578.
- Wilcke, Xander, Bloem, Peter, and de Boer, Victor. ‘The Knowledge Graph as the Default Data Model for Learning on Heterogeneous Knowledge’. 1 Jan. 2017 : 39 – 57. https://content.iospress.com/articles/data-science/ds007
- Xu B, Shi X,Zha Z, et al. Full-attention based drug drug interaction extraction exploiting user-generated content. 2018. IEEE Int Conf Bioinform Biom. 2018. IEEE Computer Society.
- Yi Z, Li S, Yu J, et al. Drug–drug interaction extraction via recurrent neural network with multiple attention layers. 2017.
- Yijia Zhang et al. “Chemical–protein interaction extraction via contextualized word representations and multihead attention.” Database, Volume 2019, 2019, baz054, https://doi.org/10.1093/database/baz054. Published: 24 May 2019
- Zhao, Zhehuan, et al. "Drug drug interaction extraction from biomedical literature using syntax convolutional neural network." Bioinformatics 32.22 (2016): 3444-3453.
- Zhao, Zhehuan, et al. "Drug drug interaction extraction from biomedical literature using syntax convolutional neural network." Bioinformatics 32.22 (2016): 3444-3453. from the literature: a review.” Briefings in Bioinformatics, 00(0), 2019, 1–19.
- Zheng W, Lin H, Luo L, et al. An attention-based effective neural model for drug-drug interactions extraction. BMC Bioinform 2017;18(1):445.
- Zhou D, Miao L,He Y. Position-aware deep multi-task learning for drug-drug interaction extraction. Artif Intell Med 2018;S0933365717306310.


