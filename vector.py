#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    document_char = torch.LongTensor([char_dict[w] for w in ex['document_char']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    question_char = torch.LongTensor([char_dict[w] for w in ex['question_char']])

    # Create extra features vector
    if len(feature_dict) > 0:
        c_features = torch.zeros(len(ex['document']), len(feature_dict))
        q_features = torch.zeros(len(ex['question']), len(feature_dict))
    else:
        c_features = None
        q_features = None

    # f_{exact_match}
    if args.use_exact_match:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                c_features[i][feature_dict['in_cased']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                c_features[i][feature_dict['in_uncased']] = 1.0
            if q_lemma and ex['clemma'][i] in q_lemma:
                c_features[i][feature_dict['in_lemma']] = 1.0

        c_words_cased = {w for w in ex['document']}
        c_words_uncased = {w.lower() for w in ex['document']}
        c_lemma = {w for w in ex['clemma']} if args.use_lemma else None
        for i in range(len(ex['question'])):
            if ex['question'][i] in c_words_cased:
                q_features[i][feature_dict['in_cased']] = 1.0
            if ex['question'][i].lower() in c_words_uncased:
                q_features[i][feature_dict['in_uncased']] = 1.0
            if c_lemma and ex['qlemma'][i] in c_lemma:
                q_features[i][feature_dict['in_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['cpos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0
        for i, w in enumerate(ex['qpos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                q_features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['cner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0
        for i, w in enumerate(ex['qner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                q_features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            c_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
        counter = Counter([w.lower() for w in ex['question']])
        l = len(ex['question'])
        for i, w in enumerate(ex['question']):
            q_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, document_char, c_features, question, question_char, q_features, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]
    
    return document, document_char, c_features, question, question_char, q_features, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 6
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    docs = [ex[0] for ex in batch]
    doc_chars = [ex[1] for ex in batch]
    c_features = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]
    question_chars = [ex[4] for ex in batch]
    q_features = [ex[5] for ex in batch]
    ids = [ex[-1] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_c = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if c_features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, c_features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(c_features[i])
    for i, c in enumerate(doc_chars):
        x1_c[i, :c.size(0)].copy_(c)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_c = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    if q_features[0] is None:
        x2_f = None
    else:
        x2_f = torch.zeros(len(questions), max_length, q_features[0].size(1))
    for i, d in enumerate(questions):
        x2[i, :d.size(0)].copy_(d)
        x2_mask[i, :d.size(0)].fill_(0)
        if x2_f is not None:
            x2_f[i, :d.size(0)].copy_(q_features[i])
    for i, c in enumerate(question_chars):
        x2_c[i, :c.size(0)].copy_(c)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][NUM_INPUTS]):
            y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
            y_e = torch.cat([ex[NUM_INPUTS+1] for ex in batch])
        else:
            y_s = [ex[NUM_INPUTS] for ex in batch]
            y_e = [ex[NUM_INPUTS+1] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, y_s, y_e, ids
