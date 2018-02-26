from collections import *
import re
def extract_lemmas(lemma_file_path): 
  with open(lemma_file_path, 'r', encoding = 'utf-8') as lemma_file: 
    lemma_file = lemma_file.read().splitlines()
    word2lemma = {}
    for line in lemma_file:
      line = re.split(r'\t+', line.rstrip('\t'))
      word2lemma[line[1]] = line[0]
  return word2lemma