#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Machine Comprehension predictor"""

import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from vector import vectorize, batchify
from model import DocReader
import utils
from spacy_tokenizer import SpacyTokenizer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenize + annotate
# ------------------------------------------------------------------------------

TOK = None

def init(options):
    global TOK
    TOK = SpacyTokenizer(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    global TOK
    return TOK.tokenize(text)

def get_annotators_for_model(model):
    annotators = set()
    if model.args.use_pos:
        annotators.add('pos')
    if model.args.use_lemma:
        annotators.add('lemma')
    if model.args.use_ner:
        annotators.add('ner')
    return annotators


# ------------------------------------------------------------------------------
# Predictor class.
# ------------------------------------------------------------------------------


class Predictor(object):
    """Load a pretrained DocReader model and predict inputs on the fly."""

    def __init__(self, model, normalize=True,
                 embedding_file=None, char_embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            normalize: squash output score to 0-1 probabilities with a softmax.
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = DocReader.load(model, normalize=normalize)

        if embedding_file:
            logger.info('Expanding dictionary...')
            utils.index_embedding_words(embedding_file)
            added_words = self.model.expand_dictionary(words)
            self.model.load_embeddings(added_words, embedding_file)
        if char_embedding_file:
            logger.info('Expanding dictionary...')
            chars = utils.index_embedding_chars(char_embedding_file)
            added_chars = self.model.expand_char_dictionary(chars)
            self.model.load_char_embeddings(added_chars, char_embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = get_annotators_for_model(self.model)

        if num_workers is None or num_workers > 0:
            self.workers = ProcessPool(
                num_workers,
                initializer=init,
                initargs=({'annotators': annotators},),
            )
        else:
            self.workers = None
            self.tokenizer = SpacyTokenizer(annotators=annotators)

    def predict(self, document, question, candidates=None, top_n=1):
        """Predict a single document - question pair."""
        results = self.predict_batch([(document, question, candidates,)], top_n)
        return results[0]

    def predict_batch(self, batch, top_n=1):
        """Predict a batch of document - question pairs."""
        documents, questions, candidates = [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if any(candidates) else None

        # Tokenize the inputs, perhaps multi-processed.
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            c_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            c_tokens = list(c_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            c_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'question_char': q_tokens[i].chars(),
                'qlemma': q_tokens[i].lemmas(),
                'qpos': q_tokens[i].pos(),
                'qner': q_tokens[i].entities(),
                'document': c_tokens[i].words(),
                'document_char': c_tokens[i].chars(),
                'clemma': c_tokens[i].lemmas(),
                'cpos': c_tokens[i].pos(),
                'cner': c_tokens[i].entities(),
            })

        # Stick document tokens in candidates for decoding
        if candidates:
            candidates = [{'input': c_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(e, self.model) for e in examples])
        s, e, score = self.model.predict(batch_exs, candidates, top_n)

        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = c_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                predictions.append((span, score[i][j]))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
