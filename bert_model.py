import logging
import os
import data_processor
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from sklearn.metrics import classification_report
import utils
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split


utils.fix_random_seeds()
SEMEVAL_HOME = os.path.join("semeval", "task9_train_pair")


logger = logging.getLogger()
logger.level = logging.ERROR


class HfBertClassifierModel(nn.Module):
    def __init__(self, n_classes, max_sentence_length=118, weights_name='bert-base-uncased'):
        super().__init__()
        self.n_classes = n_classes
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.max_sentence_length = max_sentence_length

        self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.max_sentence_length), nn.ReLU(), nn.Linear(self.max_sentence_length, self.max_sentence_length))
        self.tail = nn.Sequential(nn.Linear(self.hidden_dim, self.max_sentence_length), nn.ReLU(), nn.Linear(self.max_sentence_length, self.max_sentence_length))
        self.W = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, X):
        """Here, `X` is an np.array in which each element is a pair
        consisting of an index into the BERT embedding and a 1 or 0
        indicating whether the token is masked. The `fit` method will
        train all these parameters against a softmax objective.

        """
        indices = X[:, 0, :]
        # Type conversion, since the base class insists on
        # casting this as a FloatTensor, but we ned Long
        # for `bert`.
        indices = indices.long()
        mask = X[:, 1, :]
        (final_hidden_states, cls_output) = self.bert(
            indices, attention_mask=mask)
        head = self.head(final_hidden_states)
        tail = self.tail(final_hidden_states)
        # for the forward pass, we need to return a score and the tensor

        return self.W(cls_output)


class HfBertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, max_sentence_length, *args, **kwargs):
        # default to bert-uncased
        self.weights_name = weights_name
        self.max_sentence_length = max_sentence_length or 118
        # default to bert-uncased
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        super().__init__(*args, **kwargs)

    def define_graph(self):
        """This method is used by `fit`. We override it here to use our
        new BERT-based graph.

        """
        bert = HfBertClassifierModel(
            self.n_classes_, weights_name=self.weights_name, max_sentence_length=self.max_sentence_length)
        bert.train()
        return bert

    def encode(self, X, max_length=None):
        """The `X` is a list of strings. We use the model's tokenizer
        to get the indices and mask information.

        Returns
        -------
        list of [index, mask] pairs, where index is an int and mask
        is 0 or 1.

        """
        data = self.tokenizer.batch_encode_plus(
            X,
            max_length=max_length,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True)
        indices = data['input_ids']
        mask = data['attention_mask']
        return [[i, m] for i, m in zip(indices, mask)]


DB_dataset = data_processor.Dataset('DrugBank').from_training_data('DrugBank')
ML_dataset = data_processor.Dataset('MedLine').from_training_data('MedLine')


def create_classification_task(dataset, maxsize=None):
    '''Take a dataprocessor dataset object and return a Pandas dataframe for classification task '''

    classification_task_df = pd.DataFrame(columns=['e1_id','e1_type','e1_name','e2_id','e2_type','e2_name','sentence','ddi','label'])
    count = 0
    for doc in dataset.documents:
        for sent in doc.sentences:
            if len(sent.map)==0:
                if len(sent.entities)==1:
                    classification_task_df = classification_task_df.append({
                        'e1_id': sent.entities[0]._id,
                        'e1_type': sent.entities[0].type,
                        'e1_name': sent.entities[0].text,
                        'e2_id':"",
                        'e2_type':"",
                        'e2_name':"",
                        'sentence':sent.text,
                        'ddi':False,
                        'label':'NO_DDI'
                    },ignore_index=True)
                elif len(sent.entities)>2:
                    for (i,j)in itertools.combinations(range(len(sent.entities)),2):
                        classification_task_df = classification_task_df.append({
                            'e1_id': sent.entities[i]._id,
                            'e1_type': sent.entities[i].type,
                            'e1_name': sent.entities[i].text,
                            'e2_id':sent.entities[j]._id,
                            'e2_type':sent.entities[j].type,
                            'e2_name':sent.entities[j].text,
                            'sentence':sent.text,
                            'ddi':False,
                            'label':'NO_DDI'
                        },ignore_index=True)
            elif len(sent.map) > 0: # if there is a pair
                for i, (k, v) in enumerate(sent.map.items()):
                    for entity in sent.entities:
                        if entity._id == k:
                            e1_id = k
                            e1_type = entity.type
                            e1_name = entity.text
                        if entity._id == v[0]:
                            e2_id = v[0]
                            e2_type = entity.type
                            e2_name = entity.text
                    classification_task_df = classification_task_df.append({
                            'e1_id': e1_id,
                            'e1_type':e1_type,
                            'e1_name':e1_name,
                            'e2_id':e2_id,
                            'e2_type':e2_type,
                            'e2_name':e2_name,
                            'sentence':sent.text,
                            'ddi':True,
                            'label':v[1]
                        },ignore_index=True)
            count += 1
            if maxsize and count > maxsize:
                return classification_task_df
    return classification_task_df


def run_experiment(batch_size=16, max_iter=4, eta=0.00002, test_size=0.2, random_state=42, datasize=None):
    print('Running exp')
    classification_task_df = pd.concat([
        create_classification_task(DB_dataset, maxsize=datasize),
        create_classification_task(ML_dataset, maxsize=datasize)
    ])

    max_sentence_length = 120
    bert_experiment_1 = HfBertClassifier(
        'bert-base-uncased',
        max_sentence_length,
        batch_size=batch_size, # small batch size for use on notebook
        max_iter=max_iter,
        eta=eta)

    X_text_train, X_text_test, y_train, y_test = train_test_split(classification_task_df['sentence'], classification_task_df['label'], test_size=test_size, random_state=random_state)
    X_indices_train = bert_experiment_1.encode(X_text_train, max_length=max_sentence_length)
    X_indices_test = bert_experiment_1.encode(X_text_test, max_length=max_sentence_length)
    time = bert_experiment_1.fit(X_indices_train, y_train)
    bert_experiment_1_preds = bert_experiment_1.predict(X_indices_test)
    print(classification_report(bert_experiment_1_preds, y_test , digits=3))
    bert_experiment_1.to_pickle('BERT_exp1.pkl')


run_experiment(datasize=200)
