#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to run the reader model interactively."""

import sys
sys.path.append('.')
import torch
import code
import argparse
import logging
import prettytable
import time

from predictor import Predictor
from multiprocessing import cpu_count

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

PREDICTOR = None

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(document, question, candidates=None, top_n=1):
    t0 = time.time()
    predictions = PREDICTOR.predict(document, question, candidates, top_n)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    print(table)
    print('Time: %.4f' % (time.time() - t0))


banner = """
* WRMCQA interactive Document Reader Module *

* Repo: Mnemonic Reader (https://github.com/HKUST-KnowComp/MnemonicReader)

* Implement based on Facebook's DrQA

>>> process(document, question, candidates=None, top_n=1)
>>> usage()
"""


def usage():
    print(banner)

# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model to use')
    parser.add_argument('--embedding-file', type=str, default=None,
                        help=('Expand dictionary to use all pretrained '
                            'embeddings in this file.'))
    parser.add_argument('--char-embedding-file', type=str, default=None,
                        help=('Expand dictionary to use all pretrained '
                            'char embeddings in this file.'))
    parser.add_argument('--num-workers', type=int, default=int(cpu_count()/2),
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Specify GPU device id to use')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Do not softmax normalize output scores.')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    PREDICTOR = Predictor(
        args.model,
        normalize=not args.no_normalize,
        embedding_file=args.embedding_file,
        char_embedding_file=args.char_embedding_file,
        num_workers=args.num_workers,
    )
    if args.cuda:
        PREDICTOR.cuda()
    code.interact(banner=banner, local=locals())
