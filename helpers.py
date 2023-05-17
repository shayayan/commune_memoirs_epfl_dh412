# Helper functions for preprocessing
import pickle
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from collections import Counter
import tqdm

# across set of unique non-stop-word tokens across all our memoirs, check for misspellings of our important words, and save as dictionary
# this takes a while. run sparingly
def find_misspelled_wois(memoir_tokens_unique, wois, preserved=['thiswillliterallymatchnothing']):
    stopwords = stopwords.words('french')
    memoir_tokens_unique_nostop = [token for token in memoir_tokens_unique if token not in stopwords]
    spell = SpellChecker(language='fr')
    spell_fixer = {}
    for token in tqdm.tqdm(memoir_tokens_unique_nostop):
        if token not in preserved:
            tmp_correction = spell.correction(token)
            if tmp_correction in wois:
                spell_fixer[token] = tmp_correction
    with open('spell_fixer.pkl', 'wb') as f:
        pickle.dump(spell_fixer, f)

# check one long string (an unsplit memoir) for the most common tokens that end with .
# used for finding potential contractions
def find_potential_contractions(memoir):
    memoir = memoir.lower()
    memoir = memoir.split()
    possible_abb = []
    for token in memoir:
        if token.endswith('.'):
            possible_abb.append(token)
    counter = Counter(possible_abb)
    print(counter.most_common(100))