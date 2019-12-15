import os

from .source.data import UnicodeCharsVocabulary, Vocabulary, \
    Batcher, TokenBatcher, LMDataset, BidirectionalLMDataset

import tensorflow

path = '/Users/guozongchao/Desktop/Material/machine-leaning/nlp_demo/data/train/*'

vocobPath = '/Users/guozongchao/Desktop/Material/machine-leaning/nlp_demo/data/train/vocab.txt'

vocab_file = os.path.join(vocobPath, 'vocab.txt')

vocab = Vocabulary(vocab_file, validate_file=True)

data = BidirectionalLMDataset(path, vocab, test=test)

output = list(data.iter_batches(2, 3))

print(output)
