import torch
import re
import numpy as np
import random
import os
import flair 
import json
import codecs
import argparse
from collections import defaultdict
from flair.trainers import ModelTrainer
from typing import List
from flair.data import Sentence
from gensim.models import KeyedVectors
from flair.models import SequenceTagger
from torch.optim import Adam
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, FlairEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings
from torch.optim.sgd import SGD
from collections import defaultdict
from utils import merge_files, show_results

class W2vWordEmbeddings(TokenEmbeddings):

    def __init__(self, embeddings, static_embeddings, device, binary):
        super().__init__()
        self.name = embeddings
        self.static_embeddings = static_embeddings
        self.device = device
        self.precomputed_word_embeddings = KeyedVectors.load_word2vec_format(embeddings, binary=binary)
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub('\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '#', token.text.lower())]
                elif re.sub('\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub('\d', '0', token.text.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding).to(self.device)
                token.set_embedding(self.name, word_embedding)
        return sentences

class SingleEntityModel:
    def __init__(self, 
                corpus, 
                embeddings,
                word_dropout, 
                hidden_size, 
                rnn_layers,
                use_crf, 
        ):
        self.corpus = corpus
        self.embeddings = embeddings
        self.word_dropout = word_dropout
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.use_crf = use_crf
        
    def create_tagger(self):
        tag_dictionary = self.corpus.make_tag_dictionary(tag_type = 'ner')
        tagger: SequenceTagger = SequenceTagger(
                                    hidden_size = self.hidden_size,
                                    dropout = self.word_dropout,
                                    embeddings = self.embeddings,
                                    tag_dictionary = tag_dictionary,
                                    use_crf = self.use_crf,
                                    rnn_layers = self.rnn_layers,
                                    tag_type = 'ner'
                                )
        return tagger

class SingleEntityTrainer:
    def __init__(self, 
            corpus,
            tagger,
            entity, 
            epochs, 
            learning_rate, 
            mini_batch_size,
            optimizer,
            output_path
    ):  
        self.corpus = corpus
        self.tagger = tagger
        self.entity = entity
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.output_path = output_path

    def train(self):

        trainer: ModelTrainer = ModelTrainer(
                model = self.tagger, 
                corpus = self.corpus,
                optimizer = SGD if self.optimizer == 'SGD' else Adam)
        
        trainer.train(
            base_path = f'{self.output_path}/{self.entity}',
            learning_rate = self.learning_rate,
            train_with_dev = True,  
            embeddings_storage_mode = 'gpu',
            mini_batch_size = self.mini_batch_size,
            max_epochs = self.epochs
            ) 
    
 
             
        
if __name__ == '__main__':
    # Get parameters from JSON file.
    f = open('params.json', )
    params = json.load(f)
    device = params["device"]
    dataset = params["dataset"]

    # Select the device to use.
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device(device)
    else:
        flair.device = torch.device('cpu')

    # In this case the execution will be carried out sequentially, but depending on the available resources, the models of each entity type can be executed in parallel.
    # By default it will run sequentially, but you can send different jobs running different entity type groups.
    # Define the list of entity types. In GermEval, for simplicity and the small number of examples in some entity types, we group DERIV and PART variants.
    if dataset == 'wl':
        entities = ['Abbreviation', 'Disease', 'Finding', 'Family_Member', 'Medication', 'Body_Part', 'Procedure']
    if dataset == 'genia':
        entities = ['DNA', 'RNA',  'protein', 'cell_line', 'cell_type']
    if dataset == 'germeval':
        entities = ['ORG', 'PER',  'LOC', 'OTH', 'LOCderiv', 'OTHpart', 'deriv', 'part']


    # Next, we define the combination of embeddings to use in the Stacked Embedding Layer.
    embedding_types: List[TokenEmbeddings] = []

    if params["use_pretrained_embeddings"]: 
        if dataset == 'genia': 
            embedding_types.append(W2vWordEmbeddings('embeddings/PubMed-shuffle-win-30.bin', static_embeddings = params["static_embeddings"], device = device, binary=True))

        if dataset == 'germeval': 
            embedding_types.append(W2vWordEmbeddings('embeddings/de-crawl.txt', static_embeddings = params["static_embeddings"], device = device, binary=False))
            # You can also use WordEmbeddings('de-crawl') from flair.
        if dataset == 'wl': 
            embedding_types.append(W2vWordEmbeddings('embeddings/cwlce.vec', static_embeddings = params["static_embeddings"], device = device, binary=False))

    if params["use_char_embeddings"]:
        embedding_types.append(CharacterEmbeddings())
    
    if params["use_bert_embeddings"]:
        if dataset == 'genia':
            embedding_types.append(TransformerWordEmbeddings(
            'bert-large-cased', 
            layers = params["bert_layers"], 
            fine_tune=False, 
            layer_mean = True if params["bert_method"] == "mean" else False, 
            subtoken_pooling = 'mean'
        ))

        if dataset == 'germeval':
            embedding_types.append(TransformerWordEmbeddings(
            'bert-base-german-cased', 
            layers = params["bert_layers"], 
            fine_tune=False, 
            layer_mean = True if params["bert_method"] == "mean" else False, 
            subtoken_pooling = 'mean'
        ))

        if dataset == 'wl':
            embedding_types.append(TransformerWordEmbeddings(
            'dccuchile/bert-base-spanish-wwm-cased', 
            layers = params["bert_layers"], 
            fine_tune=False, 
            layer_mean = True if params["bert_method"] == "mean" else False, 
            subtoken_pooling = 'mean'
        ))
        
    if params["use_flair_embeddings"]:
        if dataset == 'genia':
            embedding_types.append(FlairEmbeddings('pubmed-forward'))
            embedding_types.append(FlairEmbeddings('pubmed-backward'))

        if dataset == 'germeval':
            embedding_types.append(FlairEmbeddings('de-forward'))
            embedding_types.append(FlairEmbeddings('de-backward'))
            
        if dataset == 'wl':
            embedding_types.append(FlairEmbeddings('spanish-forward'))
            embedding_types.append(FlairEmbeddings('spanish-backward'))
            # You can also use the models: 'es-clinical-forward', 'es-clinical-backward', trained by us.

    flair_embeddings: StackedEmbeddings = StackedEmbeddings(embeddings = embedding_types)

    my_dict = defaultdict(list)

    for entity in entities: 
        
        # We load the data in conll format.
        corpus = NLPTaskDataFetcher.load_column_corpus(
                                        data_folder = f'{dataset}/', 
                                        column_format = {0: 'text', 1: 'ner'},
                                        train_file = f'{entity}/{entity}_train.conll',
                                        test_file = f'{entity}/{entity}_test.conll',
                                        dev_file = f'{entity}/{entity}_dev.conll'
        )
        
        # Creating a single entity model associated with currently entity type.
        tagger = SingleEntityModel(
                            corpus = corpus, 
                            embeddings = flair_embeddings,
                            word_dropout = params["word_dropout"], 
                            hidden_size = params["hidden_size"], 
                            rnn_layers = params["rnn_layers"],
                            use_crf = params["use_crf"]
        ).create_tagger()
        
        # Training Single Entity Model
        SingleEntityTrainer(
                        corpus = corpus,
                        tagger = tagger,
                        entity = f'{entity}',
                        epochs = params["max_epochs"],
                        learning_rate = params["learning_rate"],
                        mini_batch_size = params["mini_batch_size"],
                        optimizer = params["optimizer"],
                        output_path = params["output_folder"]
        ).train()
        
    # If you are using es-clinical-forward and es-clinical-backward comment out the following lines.

    # We merge each model output file into one single output file containing nested entities and entities tagged with more than one entity type.
    merge_files(entities, params["output_folder"])

    # Finally, we print the metrics. 
    show_results(entities, params["output_folder"])   