import pickle
import sys
from nltk.tokenize import RegexpTokenizer
from utils import load_matrix
import statistics


# Given a file of words and a file of cuis, generate pickled dictionaries bi-directionally mapping cuis to words
def map_cuis_to_terms(words_file, cuis_file):
    words = []
    cuis = []
    with open(words_file) as f:
        for line in f.readlines():
            words.append(line.rstrip())
    with open(cuis_file) as f:
        for line in f.readlines():
            cuis.append(line.rstrip())
    cui_to_word = {}
    word_to_cui = {}
    for i in range(len(words)):
        word = words[i]
        cui = cuis[i]
        word_to_cui[word] = cui
        cui_to_word[cui] = word
    with open('term_to_cui.pkl', 'wb') as pkl_file:
        pickle.dump(word_to_cui, pkl_file)
    with open('cui_to_term.pkl', 'wb') as pkl_file:
        pickle.dump(cui_to_word, pkl_file)


# Get the keys, lines, and max value (for normalization) from file
def get_keys_lines_max(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
        keys = lines[0].split()
        lines = lines[1:]
        max_val = -sys.maxsize
        for line in lines:
            for val in line.split():
                try:
                    val = float(val)
                    max_val = max(max_val, val)
                except:
                    pass
        return keys, lines, max_val


# Given a UMLS similarity matrix file, generate a similarity matrix
def generate_matrix(fname):
    # Get keys, lines, and max values
    keys, lines, max_val = get_keys_lines_max(fname)
    key_to_idx = {k: v for v, k in enumerate(keys)}
    # Initialize matrix of -1's
    mat = [[-1 for _ in range(len(keys))] for _ in range(len(keys))]
    # Iterate over lines in file
    for i in range(len(lines)):
        line = lines[i].split()
        line = line[1:]
        for j in range(len(line)):
            val = float(line[j])
            # Normalize the value
            if val != -1:
                val = val / max_val
            mat[i][j] = val
    # Save matrix and key index as pickle files
    with open(fname[:-4] + '_matrix.pkl', 'wb') as mat_file:
        pickle.dump(mat, mat_file)
    with open(fname[:-4] + '_key_idx.pkl', 'wb') as pkl_file:
        pickle.dump(key_to_idx, pkl_file)


# Given a matrix, key to index mapping, and two words, return the similarity score in the matrix, if it exists
def get_from_matrix(mat, key_to_idx, word_1, word_2):
    # If the words are not in the key to index mapping, return -1
    if word_1 not in key_to_idx or word_2 not in key_to_idx:
        return -1
    # Otherwise, return the similarity score stored in the matrix
    word_1_idx = key_to_idx[word_1]
    word_2_idx = key_to_idx[word_2]
    return mat[word_1_idx][word_2_idx]


# Given a list of key to index files, list of matrix files, and mapping from CUI to term, create a combined matrix,
# where each entry i,j is the median similarity score from all matrices of word i and word j
def combine_matrices(key_idx_files, mat_files, cui_to_term_f="./similarities/cui_to_term.pkl"):
    # Initialize set of unique keys
    unique_keys = set()
    key_idxs = []
    mats = []
    # Load all key to index mappings and matrices
    for i in range(len(key_idx_files)):
        keys_i, mat_i = load_matrix(key_idx_files[i], mat_files[i], map_cui_to_term=True, cui_to_term_f=cui_to_term_f)
        key_to_idx_i = {k: v for v, k in enumerate(keys_i)}
        unique_keys.update(set(keys_i))
        mats.append(mat_i)
        key_idxs.append(key_to_idx_i)

    # Create master list of unique keys and master mapping
    unique_keys = list(unique_keys)
    key_to_idx = {k: v for v, k in enumerate(unique_keys)}

    # Initialize a matrix of -1's
    mat = [[-1 for _ in range(len(unique_keys))] for _ in range(len(unique_keys))]
    # Iterate over all word pairs
    for i in range(len(unique_keys)):
        word_1 = unique_keys[i]
        for j in range(len(unique_keys)):
            word_2 = unique_keys[j]
            # Get similarity values from all matrices
            vals = [get_from_matrix(mats[k], key_idxs[k], word_1, word_2) for k in range(len(mats))]
            # Get values that are not -1
            non_neg_vals = [val for val in vals if val >= 0]
            # If there are no non-negative values, insert -1 in the matrix
            if not non_neg_vals:
                mat[i][j] = -1
            # Otherwise, insert the median of the values into the matrix
            else:
                mat[i][j] = statistics.median(non_neg_vals)
    # Save to pickle files
    with open('test_matrix.pkl', 'wb') as mat_file:
        pickle.dump(mat, mat_file)
    with open('test_key_idx.pkl', 'wb') as pkl_file:
        pickle.dump(key_to_idx, pkl_file)


# Given a file, tokenize and return list of cleaned words
def process_file(file_name):
    file_text = []
    tokenizer = RegexpTokenizer(r'\w+')
    with open(file_name, "r", encoding='utf-8', errors='ignore') as file:
        for line in file.readlines():
            try:
                words = tokenizer.tokenize(line)
                for word in words:
                    word = word.strip().lower()
                    # If the word has at least 3 characters and at least one of them is a letter, append it
                    if len(word) > 2 and any(c.isalpha() for c in word):
                        file_text.append(word)
            except:
                pass
        for word in file_text:
            print(word)
        return file_text


if __name__ == "__main__":
    combine_matrices(
        ["./similarities/pickle/lch_key_idx.pkl", "./similarities/pickle/wup_key_idx.pkl", "./similarities/pickle/nam_key_idx.pkl"],
        ["./similarities/pickle/lch_matrix.pkl", "./similarities/pickle/wup_matrix.pkl", "./similarities/pickle/nam_matrix.pkl"]
    )
