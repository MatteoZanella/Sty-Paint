import pickle
import os
import yaml
import numpy as np


def make_dir_tree(base_path):
    os.makedirs(base_path, exist_ok=True)

    types = ['greedy', 'lkh']
    tmp = ['index', 'videos']

    for t in types:
        os.makedirs(os.path.join(base_path, t), exist_ok=True)
        for s in tmp:
            os.makedirs(os.path.join(base_path, t, s), exist_ok=True)
        if t == 'lkh':
            os.makedirs(os.path.join(base_path, t, 'lkh_files'), exist_ok=True)

    print('Directory Tree created')

def save_pickle(obj, path):
    path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj



########################################################################################################################
class LKHConfig():
    def __init__(self, default_config_path, name, num_nodes, output_path):

        # Load tempalte
        self.laod_lkh_files(default_config_path)

        # Specific parameters
        self.name = name
        self.num_nodes = num_nodes
        self.output_path = os.path.join(output_path, name)
        # add parameters
        self.add_params()


    def laod_lkh_files(self, path):
        with open(os.path.join(path, 'problem.yaml'), 'r') as f:
            self.problem_file = yaml.safe_load(f)
        with open(os.path.join(path, 'conf.yaml'), 'r') as f:
            self.conf_file = yaml.safe_load(f)

    def add_params(self):
        # Update problem file
        self.problem_file['NAME'] = '{}.sop'.format(self.name)
        self.problem_file['DIMENSION'] = self.num_nodes
        self.problem_file['EDGE_WEIGHT_SECTION'] = self.num_nodes

        # Update config file
        self.conf_file['PROBLEM_FILE'] = os.path.join(self.output_path + '.sop')
        self.conf_file['TOUR_FILE'] = os.path.join(self.output_path + '_solution.txt')

    def parse_files(self, cost_matrix):
        # Write Problem file
        with open(os.path.join(self.output_path + '.sop'), 'w') as f:
            f.write(f"NAME:{self.problem_file['NAME']}\n")
            f.write(f"TYPE:{self.problem_file['TYPE']}\n")
            f.write(f"COMMENT:{self.problem_file['COMMENT']}\n")
            f.write(f"DIMENSION:{self.problem_file['DIMENSION']}\n")
            f.write(f"EDGE_WEIGHT_TYPE:{self.problem_file['EDGE_WEIGHT_TYPE']}\n")
            f.write(f"EDGE_WEIGHT_FORMAT:{self.problem_file['EDGE_WEIGHT_FORMAT']}\n")
            f.write(f"EDGE_WEIGHT_SECTION\n{self.problem_file['EDGE_WEIGHT_SECTION']}\n")

            # save the weight matrix
            np.savetxt(f, cost_matrix, delimiter='\t', fmt='%d')
            f.write('\nEOF')

        # Write configuration file
        self.conf_file_path = os.path.join(self.output_path + '.par')
        with open(self.conf_file_path, 'w') as f:
            if self.conf_file['SPECIAL']:
                f.write('SPECIAL\n')

            f.write(f"PROBLEM_FILE = {self.conf_file['PROBLEM_FILE']}\n")
            f.write(f"TOUR_FILE = {self.conf_file['TOUR_FILE']}\n")
            f.write(f"TIME_LIMIT = {self.conf_file['TIME_LIMIT']}\n")
            f.write(f"RUNS = {self.conf_file['RUNS']}")

