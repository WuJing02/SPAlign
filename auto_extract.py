import os
import argparse

import pcode.utils.op_files as op_files
from pcode.tools.show_results import load_raw_info_from_experiments

def get_args():
    parser = argparse.ArgumentParser(description="Extract results.")

    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_name", type=str, default="summary.pickle")

    args = parser.parse_args()

    check_args(args)
    return args

def check_args(args):
    assert args.in_dir is not None

    args.out_path = os.path.join(args.in_dir, args.out_name)

def main(args):
    op_files.write_pickle(load_raw_info_from_experiments(args.in_dir), args.out_path)

if __name__ == "__main__":
    args = get_args()

    main(args)
