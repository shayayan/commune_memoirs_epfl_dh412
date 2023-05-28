# Helper functions for preprocessing
import pickle
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from collections import Counter
import tqdm
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from numpy import dot
from numpy.linalg import norm
import spacy
import pandas as pd
import re

# Simple cosine similarity
def cos_sim(v1,v2):
    if norm(v1) == 0 or norm(v2) == 0:
        print("empty vector in cos_sim, something wrong")
        error
    return dot(v1,v2)/(norm(v1)*norm(v2))

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

# extract the context surrounding certain words (or phrases) of interest
# method can be "token", "character", "<sb>", or "<pb>". The first two must be followed by an integer in the arguments.
# character, <sb>, <pb> are way faster than token. 
# character is probably the most appropriate extraction method given the unreliability of sb and pb tokens.
def extract_woi_context(commune_memoirs, wois, method, count = 0, pull_synonyms = False):

    if pull_synonyms == True:
        tmp = []
        for word in wois:
            tmp.extend(pull_lemmatized_synonyms(word))
        wois = list(set(tmp + wois))

    woi_context_extraction = pd.DataFrame(columns=['filename', 'memoir_len_sans_sb_pb', 'bias', 'woi', 'woi_location', 'extraction_method', 'text'])

    for i, row in tqdm(commune_memoirs.iterrows()):

        filename = row[0]

        # commented out because this will corrupt lengths by token count
        #if filename == "du_camp_1.txt" or filename == "du_camp_2.txt":
        #    filename = "du_camp.txt"
        #if filename == "arnould_1.txt" or filename == "arnould_2.txt" or filename == "arnould_3.txt":
        #    filename = "arnould.txt"
        #if filename == "cluseret_1.txt" or filename == "cluseret_2.txt" or filename == "cluseret_3.txt":
        #    filename = "cluseret.txt"
        #if filename == "da_costa_1.txt" or filename == "da_costa_2.txt" or filename == "da_costa_3.txt":
        #    filename = "da_costa.txt"

        bias = row[1]
        text = row[2]
        
        memoir_len_sans_sb_pb = len([token for token in text.split() if token not in ["<sb>","<pb>"]])

        text_lower = text.lower()
        for woi in wois:
            if woi.lower() in text_lower:
                indices_object = re.finditer(pattern=r'\b{}\b'.format(re.escape(woi.lower())), string=text_lower)
                indices = [index.start() for index in indices_object]
                for location in indices:
                    if method == "token" or method == "character":
                        method_s = str(method) + ", " + str(count) + " either side"
                        window_half = count
                        if method == "token":
                            pre_tokens = ' '.join(text_lower[:location].split()[-window_half:])
                            post_tokens = ' '.join(text_lower[location:].split()[:window_half+len(woi.split())])
                            tmp_sequence = pre_tokens + " " + post_tokens
                            woi_location = len(pre_tokens) + 1
                        if method == "character":
                            pre_string = text_lower[:location][-window_half:]
                            post_string = text_lower[location:][:window_half+len(woi)]
                            tmp_sequence = pre_string + post_string
                            woi_location = 0
                            woi_location = len(pre_string)
                    elif method == "<sb>" or method == "<pb>":
                        method_s = method
                        pre_string = text_lower[text_lower[:location].rfind(method):location]
                        if "<pb>" in pre_string and method == "<sb>":
                            pre_string = pre_string[pre_string.rfind("<pb>"):]
                        post_string_tmp = text_lower[location:]
                        post_string = post_string_tmp[:post_string_tmp.find(method) + 4]
                        if "<pb>" in post_string and method == "<sb>":
                            post_string = post_string[:post_string.find("<pb>")]
                        tmp_sequence = pre_string + post_string
                        woi_location = 0
                        woi_location = len(pre_string)
                    woi_context_extraction.loc[len(woi_context_extraction)] = [filename, memoir_len_sans_sb_pb, bias, woi, woi_location, method_s, tmp_sequence]
    
    return woi_context_extraction

# part of metaphor recognition
# construct set of synonyms, hypernyms, and inflections from WordNet for a single word
# pass in matrix in preserve words' candidate sets in memory (helps if looping over lots of text)
# inflecteur is passed in because it takes a while to initialize it
# 'candidate set' from https://aura.abdn.ac.uk/bitstream/handle/2164/10781/P18_1113.pdf?sequence=1
def find_candidate_set(word, inflecteur, dp_matrix = {}):
    if word in dp_matrix.keys():
        return dp_matrix[word], dp_matrix
    else:
        synsets = wn.synsets(word, lang='fra')
        synonyms = []
        hypernyms = []
        for synset in synsets:
            synonyms.extend(synset.lemma_names(lang='fra'))
            hypernyms.extend(synset.hypernyms())
        hypernym_names = []
        for hypernym in hypernyms:
            hypernym_lemmas = hypernym.lemmas(lang='fra')
            hypernym_names.extend([lemma.name() for lemma in hypernym_lemmas])
        hypernym_names = list(set(hypernym_names))
        words_to_inflect = synonyms + [word] + hypernym_names
        words_to_inflect = list(set([t.lower() for t in words_to_inflect]))
        inflections = []
        for word in words_to_inflect:
            try:
                inflections = inflections + inflecteur.get_word_form(word)["lemma"].tolist()
            except:
                pass
        candidate_set = list(set(words_to_inflect + inflections))
        dp_matrix[word] = candidate_set
        return candidate_set, dp_matrix

# part of metaphor recognition
# compare candidate set tokens to an average embedding to find the best candidate
# spacy model passed in
def find_best_candidate(candidate_set, sans_t_avg_embedding, nlp):
    max_sim = -2
    best_candidate = None
    for candidate in candidate_set:
        if nlp(candidate).has_vector == True:
            curr_sim = cos_sim(nlp(candidate).vector, sans_t_avg_embedding)
            if curr_sim > max_sim:
                max_sim = curr_sim
                best_candidate = candidate
    return best_candidate

# synonym extractor (based on wordnet)
def pull_lemmatized_synonyms(word):
    synsets = wn.synsets("ivre", lang='fra')
    synonyms = []
    for synset in synsets:
        synonyms.extend(synset.lemma_names(lang='fra'))
    return list(set(synonyms + [word]))

# list_to_median - for making sentence embedding conclusions
def median(lst):
    if len(lst) != 0:
        return np.median(lst)
    else:
        return 99
def min(lst):
    if len(lst) != 0:
        return np.min(lst)
    else:
        return 99
def pctile(lst):
    if len(lst) != 0:
        return np.percentile(lst, 10)
    else:
        return 99