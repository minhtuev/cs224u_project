import logging
import os
import data_processor
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from sklearn.metrics import classification_report
import utils
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import torch.utils.data
from utils import progress_bar
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import random

utils.fix_random_seeds()
SEMEVAL_HOME = os.path.join("semeval", "task9_train_pair")


logger = logging.getLogger()
logger.level = logging.ERROR


class HfBertClassifierModel(nn.Module):
    def __init__(self, max_sentence_length=120, weights_name='bert-base-cased'):
        super().__init__()
        self.weights_name = weights_name
        self.bert = BertModel.from_pretrained(self.weights_name)
        self.hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.max_sentence_length = max_sentence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # dim : max length x max length
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.max_sentence_length),
            nn.ReLU(),
            nn.Linear(self.max_sentence_length, self.hidden_dim))

        self.tail = nn.Sequential(
            nn.Linear(self.hidden_dim, self.max_sentence_length),
            nn.ReLU(),
            nn.Linear(self.max_sentence_length, self.hidden_dim))

        # initialize a random tensor
        self.L = torch.randn(self.hidden_dim, self.hidden_dim).to(self.device)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory: directory to which to save.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def bilinear(self, head, tail):
        lin = torch.mm(head.reshape([-1, self.hidden_dim]), self.L).reshape([-1, self.max_sentence_length, self.hidden_dim])
        bi_lin = torch.matmul(lin, tail.reshape([-1, self.hidden_dim, self.max_sentence_length]))
        return torch.sigmoid(bi_lin)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,):
        """Here, `X` is an np.array in which each element is a pair
        consisting of an index into the BERT embedding and a 1 or 0
        indicating whether the token is masked. The `fit` method will
        train all these parameters against a softmax objective.

        The shape of X is batchsize x 2 x max_sentence_length
        """
        (final_hidden_states, cls_output) = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        head = self.head(final_hidden_states)
        tail = self.tail(final_hidden_states)
        # for the forward pass, we need to return a score and the tensor
        output = self.bilinear(head, tail)
        return output


class HfBertClassifier(TorchShallowNeuralClassifier):
    def __init__(self, weights_name, max_sentence_length, *args, **kwargs):
        # default to bert-uncased
        self.weights_name = weights_name
        self.max_sentence_length = max_sentence_length or 118
        # default to bert-uncased
        self.tokenizer = BertTokenizer.from_pretrained(self.weights_name)
        self.model: HfBertClassifierModel = None
        super().__init__(*args, **kwargs)


    def fit(self, X, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('dev_iter', 10)
        # Data prep:
        all_input_ids = torch.tensor([f.input_ids for f in X], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in X], dtype=torch.long).to(self.device)
        all_segment_ids = torch.tensor([f.segment_ids for f in X], dtype=torch.long).to(self.device)
        all_rels = torch.tensor([f.rels for f in X], dtype=torch.float).to(self.device)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_rels)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=False) #cannot pin for GPU tensors

        # Graph:
        if not self.warm_start or not hasattr(self, "model"):
            self.model = self.define_graph()
            self.opt = self.optimizer(
                self.model.parameters(),
                lr=self.eta,
                weight_decay=self.l2_strength)

        self.model.to(self.device)
        self.model.train()
        # Optimization:
        loss = nn.BCELoss()
        global_step = 0
        # Train:
        with tqdm(total=self.max_iter) as pbar:
            for iteration in range(1, self.max_iter+1):
                epoch_error = 0.0
                for batch in dataloader:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    inputs["token_type_ids"] = (batch[2])
                    batch_preds = self.model(**inputs)
                    err = loss(batch_preds.reshape([-1, self.max_sentence_length]),
                               batch[3].reshape([-1, self.max_sentence_length]))
                    epoch_error += err.item()
                    self.opt.zero_grad()
                    err.backward()
                    self.opt.step()
                    global_step += 1

                    # Save model checkpoint
                    if global_step % 100 == 0:
                        output_dir = os.path.join('checkpoints', "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        torch.save(self.opt.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                # Incremental predictions where possible:
                if X_dev is not None and iteration > 0 and iteration % dev_iter == 0:
                    preds = self.predict(X_dev)
                    print(classification_report(preds.reshape(-1),
                                                np.asarray([item.rels for item in X_dev]).reshape(-1),
                                                digits=3))
                    self.model.train()

                self.errors.append(epoch_error)

                print("Finished epoch {} of {}; error is {}".format(
                        iteration, self.max_iter, epoch_error))

                progress_bar(
                    "Finished epoch {} of {}; error is {}".format(
                        iteration, self.max_iter, epoch_error))
                pbar.update(1)
        pbar.close()
        return self

    def define_graph(self):
        """This method is used by `fit`. We override it here to use our
        new BERT-based graph.

        """
        bert = HfBertClassifierModel(weights_name=self.weights_name, max_sentence_length=self.max_sentence_length)
        bert.train()
        return bert

    def predict(self, X):
        """Predicted probabilities for the examples in `X`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array with shape (len(X), self.n_classes_)

        """
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            # Data prep:
            all_input_ids = torch.tensor([f.input_ids for f in X], dtype=torch.long).to(self.device)
            all_input_mask = torch.tensor([f.input_mask for f in X], dtype=torch.long).to(self.device)
            all_segment_ids = torch.tensor([f.segment_ids for f in X], dtype=torch.long).to(self.device)
            all_rels = torch.tensor([f.rels for f in X], dtype=torch.float).to(self.device)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_rels)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True,
                pin_memory=False ) #cannot pin GPU tensors

            preds = []
            for batch in dataloader:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                inputs["token_type_ids"] = (batch[2])
                batch_preds = self.model(**inputs)
                preds.append(torch.round(batch_preds).to(dtype=torch.int).cpu().numpy())

            result = preds[0]
            for i in range(len(preds) - 1):
                result = np.vstack((result, preds[i + 1]))
            return result


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

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, rels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.rels = rels


def convert_examples_to_features(
    dataset,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    datasize=None
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []

    counter = 0

    for doc in dataset.documents:
        for sent in doc.sentences:
            counter += 1

            if datasize and counter > datasize:
                break

            if sent.text == '':
                continue

            word_tokens = sent.text.split()
            relation_pairs = [[] for _ in range(len(word_tokens))]

            # fill with no relation
            for i in range(len(word_tokens)):
                for j in range(len(word_tokens)):
                    relation_pairs[i].append(0)

            entity_map = {}
            for entity in sent.entities:
                if len(entity.char_offset.split('-')) == 2:
                    entity_map[entity._id] = entity.char_offset

            for key in sent.map:

                if key not in entity_map:
                    print("frand")
                    continue

                source_start, source_end = [int(x) for x in entity_map[key].split('-')]
                dst_start, dst_end = [int(x) for x in entity_map[sent.map[key][0]].split('-')]

                source_indices = []
                dst_indices = []
                num_source_source = len(sent.text[source_start: source_end + 1].split())
                num_dst_source = len(sent.text[dst_start: dst_end + 1].split())

                curr_span = 0
                for i, token in enumerate(word_tokens):
                    if curr_span == source_start:
                        source_indices.append(i)
                        for _ in range(num_source_source - 1):
                            i += 1
                            source_indices.append(i)
                        break
                    else:
                        curr_span += len(token) + 1

                curr_span = 0
                for i, token in enumerate(word_tokens):
                    if curr_span == dst_start:
                        dst_indices.append(i)
                        for _ in range(num_dst_source - 1):
                            curr_span += 1
                            dst_indices.append(i)
                        break
                    else:
                        curr_span += len(token) + 1

                for i in source_indices:
                    for j in dst_indices:
                        relation_pairs[i][j] = 1

            relation_pairs = np.asarray(relation_pairs)
            relation_pairs_tokenized = []

            tokens = [cls_token]
            for i, word_i in enumerate(word_tokens):
                src_word_tokens = tokenizer.tokenize(word_i)
                for tok in src_word_tokens:
                    relation_pairs_tokenized.append(relation_pairs[i])
                    tokens.append(tok)

            if len(tokens) > max_seq_length - 2:
                continue

            relation_pairs_tokenized = np.asarray(relation_pairs_tokenized)
            relation_pairs_tokenized_2 = relation_pairs_tokenized[:,0]

            for j, word_j in enumerate(word_tokens):
                dst_word_tokens = tokenizer.tokenize(word_j)
                for _ in range(len(dst_word_tokens)):
                    relation_pairs_tokenized_2 = np.column_stack((
                        relation_pairs_tokenized_2, relation_pairs_tokenized[:,j]))

            relation_pairs_final = relation_pairs_tokenized_2[:,1:]

            # inconsistency with the new version of the Bert Transformer
            try:
                special_tokens_count = tokenizer.num_added_tokens()
            except AttributeError:
                special_tokens_count = tokenizer.num_special_tokens_to_add()

            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                relation_pairs_final = relation_pairs_final[
                                       : (max_seq_length - special_tokens_count),
                                       : (max_seq_length - special_tokens_count)]

            segment_ids = [sequence_a_segment_id] * len(tokens)
            segment_ids = [cls_token_segment_id] + segment_ids
            tokens += [sep_token]

            # add rel for [CLS] token
            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            relation_pairs_final = np.row_stack((rel, relation_pairs_final))
            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            relation_pairs_final = np.column_stack((rel, relation_pairs_final))

            # add rel for [SEP] token
            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            relation_pairs_final = np.row_stack((relation_pairs_final, rel))
            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            relation_pairs_final = np.column_stack((relation_pairs_final, rel))

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            for _ in range(padding_length):
                relation_pairs_final = np.row_stack((rel, relation_pairs_final))

            rel = [0 for _ in range(relation_pairs_final.shape[0])]
            for _ in range(padding_length):
                relation_pairs_final = np.column_stack((rel, relation_pairs_final))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(relation_pairs_final) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids, input_mask=input_mask,
                              segment_ids=segment_ids, rels=relation_pairs_final)
            )

    return features


def run_experiment(batch_size=32, max_iter=4, eta=0.00002, random_state=42, datasize=None, bert_model='bert-base-cased', max_sentence_length = 120):
    print('Running exp')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    examples = convert_examples_to_features(DB_dataset, max_sentence_length, tokenizer, datasize=datasize)
    examples.extend(convert_examples_to_features(ML_dataset, max_sentence_length, tokenizer, datasize=datasize))
    random.seed(random_state)
    random.shuffle(examples)

    train_index = int(0.7 * len(examples))
    dev_index = int(0.8 * len(examples))

    train = examples[:train_index]
    dev = examples[train_index:dev_index]
    test = examples[dev_index:]

    bert_experiment_1 = HfBertClassifier(
        bert_model,
        max_sentence_length,
        batch_size=batch_size, # small batch size for use on notebook
        max_iter=max_iter,
        device=device,
        eta=eta)

    time = bert_experiment_1.fit(train, **{'X_dev': dev})
    bert_experiment_1_preds = bert_experiment_1.predict(test)

    print(classification_report(bert_experiment_1_preds.reshape(-1),
                                np.asarray([item.rels for item in test]).reshape(-1),
                                digits=3))


run_experiment(datasize=100)
#run_experiment()
