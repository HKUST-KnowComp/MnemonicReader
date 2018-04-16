#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the R-Net based reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class R_Net(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args, normalize=True):
        super(R_Net, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(args.char_size,
                                      args.char_embedding_dim,
                                      padding_idx=0)

        # Char rnn to generate char features
        self.char_rnn = layers.StackedBRNN(
            input_size=args.char_embedding_dim,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_input_size = args.embedding_dim + args.char_hidden_size * 2

        # Encoder
        self.encode_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=args.concat_rnn_layers,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # Output sizes of rnn encoder
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        
        # Gated-attention-based RNN of the whole question
        self.question_attn = layers.SeqAttnMatch(question_hidden_size, identity=False)
        self.question_attn_gate = layers.Gate(doc_hidden_size + question_hidden_size)
        self.question_attn_rnn = layers.StackedBRNN(
            input_size=doc_hidden_size + question_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        question_attn_hidden_size = 2 * args.hidden_size

        # Self-matching-attention-baed RNN of the whole doc
        self.doc_self_attn = layers.SelfAttnMatch(question_attn_hidden_size, identity=False)
        self.doc_self_attn_gate = layers.Gate(question_attn_hidden_size + question_attn_hidden_size)
        self.doc_self_attn_rnn = layers.StackedBRNN(
            input_size=question_attn_hidden_size + question_attn_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_self_attn_hidden_size = 2 * args.hidden_size

        self.doc_self_attn_rnn2 = layers.StackedBRNN(
            input_size=doc_self_attn_hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        self.ptr_net = layers.PointerNetwork(
            x_size = doc_self_attn_hidden_size, 
            y_size = question_hidden_size, 
            hidden_size = args.hidden_size, 
            dropout_rate=args.dropout_rnn,
            cell_type=nn.GRUCell,
            normalize=normalize
        )

    def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        x1_c_emb = self.char_embedding(x1_c)
        x2_c_emb = self.char_embedding(x2_c)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
            x1_c_emb = F.dropout(x1_c_emb, p=self.args.dropout_emb, training=self.training)
            x2_c_emb = F.dropout(x2_c_emb, p=self.args.dropout_emb, training=self.training)

        # Generate char features
        x1_c_features = self.char_rnn(x1_c_emb, x1_mask)
        x2_c_features = self.char_rnn(x2_c_emb, x2_mask)

        # Combine input
        crnn_input = [x1_emb, x1_c_features]
        qrnn_input = [x2_emb, x2_c_features]

        # Encode document with RNN
        c = self.encode_rnn(torch.cat(crnn_input, 2), x1_mask)
        
        # Encode question with RNN
        q = self.encode_rnn(torch.cat(qrnn_input, 2), x2_mask)

        # Match questions to docs
        question_attn_hiddens = self.question_attn(c, q, x2_mask)
        rnn_input = self.question_attn_gate(torch.cat([c, question_attn_hiddens], 2))
        c = self.question_attn_rnn(rnn_input, x1_mask)

        # Match documents to themselves
        doc_self_attn_hiddens = self.doc_self_attn(c, x1_mask)
        rnn_input = self.doc_self_attn_gate(torch.cat([c, doc_self_attn_hiddens], 2))
        c = self.doc_self_attn_rnn(rnn_input, x1_mask)
        c = self.doc_self_attn_rnn2(c, x1_mask)

        # Predict
        start_scores, end_scores = self.ptr_net(c, q, x1_mask, x2_mask)
        
        return start_scores, end_scores
