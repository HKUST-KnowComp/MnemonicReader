#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the Mnemonic Reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable


# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class MnemonicReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args, normalize=True):
        super(MnemonicReader, self).__init__()
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

        doc_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.num_features

        # Encoder
        self.encoding_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_hidden_size = 2 * args.hidden_size
        
        # Interactive aligning, self aligning and aggregating
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(args.hop):
            # interactive aligner
            self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
            self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # aggregating
            self.aggregate_rnns.append(
                layers.StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=args.hidden_size,
                    num_layers=1,
                    dropout_rate=args.dropout_rnn,
                    dropout_output=args.dropout_rnn_output,
                    concat_layers=False,
                    rnn_type=self.RNN_TYPES[args.rnn_type],
                    padding=args.rnn_padding,
                )
            )

        # Memmory-based Answer Pointer
        self.mem_ans_ptr = layers.MemoryAnsPointer(
            x_size=2*args.hidden_size, 
            y_size=2*args.hidden_size, 
            hidden_size=args.hidden_size, 
            hop=args.hop,
            dropout_rate=args.dropout_rnn,
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
        # Add manual features
        if self.args.num_features > 0:
            crnn_input.append(x1_f)
            qrnn_input.append(x2_f)

        # Encode document with RNN
        c = self.encoding_rnn(torch.cat(crnn_input, 2), x1_mask)
        
        # Encode question with RNN
        q = self.encoding_rnn(torch.cat(qrnn_input, 2), x2_mask)

        # Align and aggregate
        c_check = c
        for i in range(self.args.hop):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)

        # Predict
        start_scores, end_scores = self.mem_ans_ptr.forward(c_check, q, x1_mask, x2_mask)
        
        return start_scores, end_scores