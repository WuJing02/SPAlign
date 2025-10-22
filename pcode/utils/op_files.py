
import os
import json
import pickle
from os.path import exists
from six.moves import cPickle

def read_text_withoutsplit(path):
    
    with open(path, "r") as f:
        return f.read()

def read_txt(path):
    
    with open(path, "r") as f:
        return f.read().splitlines()

def read_json(path):
    
    with open(path, "r") as f:
        return json.load(f)

def write_txt(data, out_path, type="w"):
    
    with open(out_path, type) as f:
        f.write(data)

def load_pickle(path):
    
    with open(path, "rb") as handle:
        return pickle.load(handle)

def write_pickle(data, path):
    
    print("write --> data to path: {}\n".format(path))
    with open(path, "wb") as handle:
        pickle.dump(data, handle)

def load_cpickle(path):
    
    with open(path, "rb") as handle:
        return cPickle.load(handle)

def write_cpickle(data, path):
    
    print("write --> data to path: {}\n".format(path))
    with open(path, "wb") as handle:
        cPickle.dump(data, handle)

def output_string(data, path_output, delimiter="\n"):
    
    os.remove(path_output) if exists(path_output) else None

    for d in data:
        try:
            write_txt(d + delimiter, path_output, "a")
        except:
            print(d)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False
