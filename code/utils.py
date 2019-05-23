import pickle
from scipy.stats.stats import pearsonr
import numpy as np

# Load matrix and index files from pkl
def load_matrix(idx_file, matrix_file, map_cui_to_term=False, cui_to_term_f=None):
    with open(idx_file, "rb") as f:
        cui_idx = pickle.load(f)
        if map_cui_to_term and cui_to_term_f is not None:
            with open(cui_to_term_f, "rb") as f:
                cui_to_term = pickle.load(f)
                term_idx = {}
                for cui in cui_idx:
                    term_idx[cui_to_term[cui]] = cui_idx[cui]
                sorted_keys = sorted(term_idx, key=term_idx.get)
        else:
            sorted_keys = sorted(cui_idx, key=cui_idx.get)
    with open(matrix_file, "rb") as f:
        matrix = pickle.load(f)
    return sorted_keys, matrix


# Read dataset 1 for evaluation
def read_ds_1(f_name):
    word_pairs = []
    phys = []
    expert = []
    with open(f_name, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = [x.strip() for x in line.split(",")]
            word_pairs.append((line[0], line[1]))
            phys.append(line[2])
            expert.append(line[3])
    return word_pairs, phys, expert


# Read dataset 2 for evaluation
def read_ds_2(f_name):
    word_pairs = []
    human = []
    with open(f_name, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = [x.strip() for x in line.split(",")]
            word_pairs.append((line[0], line[1]))
            human.append(line[2])
    return word_pairs, human


# Write an evaluation report (csv), computing correlations
def write_report(f_name, similarities_1, similarities_2, unused_pairs):
    phys_cor = np.array([tup[1] for tup in similarities_1]).astype(np.float)
    expert_cor = np.array([tup[2] for tup in similarities_1]).astype(np.float)
    sim_1_cor = np.array([tup[3] for tup in similarities_1]).astype(np.float)
    human_cor = np.array([tup[1] for tup in similarities_2]).astype(np.float)
    sim_2_cor = np.array([tup[2] for tup in similarities_2]).astype(np.float)
    with open(f_name, "w") as f:
        f.write("W1,W2,Phys.,Expert,Human,Similarity\n")
        for tup in similarities_1:
            line = "{},{},{},{},{},{}\n".format(tup[0][0], tup[0][1], tup[1], tup[2], "NA", tup[3])
            print("line:", line)
            f.write(line)
        for tup in similarities_2:
            line = "{},{},{},{},{},{}\n".format(tup[0][0], tup[0][1], "NA", "NA", tup[1], tup[2])
            f.write(line)
        f.write("Physician Correlation: {}\n".format(pearsonr(phys_cor, sim_1_cor)))
        f.write("Expert Correlation: {}\n".format(pearsonr(expert_cor, sim_1_cor)))
        f.write("Human Correlation: {}\n".format(pearsonr(human_cor, sim_2_cor)))
        f.write("Unused Pairs:\n")
        line = ""
        for pair in unused_pairs:
            line += str(pair[0]) + " " + str(pair[1]) + ","
        f.write(line[:-1])


# Given a phrase, return cleaned list of words
def clean_phrase(phrase):
    return [w.strip().lower().encode('utf-8') for w in phrase.split()]


# View matrix statistics for debugging purposes
def view_matrix(key_file, mat_file):
    keys, mat = load_matrix(key_file, mat_file)
    print("there are", len(keys), "unique words in the matrix")
    print("keys:", keys)
    mat = np.array(mat)
    print(mat)
    unique, counts = np.unique(mat, return_counts=True)
    counts = dict(zip(unique, counts))
    num_neg_one = counts[-1.0]
    total = mat.size
    not_neg_one = total - num_neg_one
    print("there are", num_neg_one, "-1's and", not_neg_one, "not -1's")
    print("there are", total, "elements in total")
    print(counts)


if __name__ == "__main__":
    view_matrix("./similarities/pickle/combined_3_key_idx.pkl", "./similarities/pickle/combined_3_matrix.pkl")