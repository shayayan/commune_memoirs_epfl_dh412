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

# extract the context surrounding certain words (or phrases) of interest
# method can be "token", "character", "<sb>", or "<pb>". The first two must be followed by an integer in the arguments.
# character, <sb>, <pb> are way faster than token. 
# character is probably the most appropriate extraction method given the unreliability of sb and pb tokens.
def extract_woi_context(commune_memoirs, wois, method, count = 0):

    woi_context_extraction = pd.DataFrame(columns=['filename', 'memoir_len_sans_sb_pb', 'bias', 'woi', 'woi_location', 'extraction_method', 'text'])

    for i, row in tqdm(commune_memoirs.iterrows()):

        filename = row[0]

        if filename == "du_camp_1.txt" or filename == "du_camp_2.txt":
            filename = "du_camp.txt"
        if filename == "arnould_1.txt" or filename == "arnould_2.txt" or filename == "arnould_3.txt":
            filename = "arnould.txt"
        if filename == "cluseret_1.txt" or filename == "cluseret_2.txt" or filename == "cluseret_3.txt":
            filename = "cluseret.txt"
        if filename == "da_costa_1.txt" or filename == "da_costa_2.txt" or filename == "da_costa_3.txt":
            filename = "da_costa.txt"

        bias = row[1]
        text = row[2]
        
        memoir_len_sans_sb_pb = len([token for token in text.split() if token not in ["<sb>","<pb>"]])

        text_lower = text.lower()
        for woi in wois:
            if woi.lower() in text_lower:
                indices_object = re.finditer(pattern=woi, string=text_lower)
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
                        post_string_tmp = text_lower[location:]
                        post_string = post_string_tmp[:post_string_tmp.find(method) + 4]
                        tmp_sequence = pre_string + post_string
                        woi_location = 0
                        woi_location = len(pre_string)
                    woi_context_extraction.loc[len(woi_context_extraction)] = [filename, memoir_len_sans_sb_pb, bias, woi, woi_location, method_s, tmp_sequence]
    
    return woi_context_extraction