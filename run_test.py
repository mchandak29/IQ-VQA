# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from torch.utils.data import DataLoader
from train_model.dataset_utils import prepare_test_data_set,prepare_eval_data_set
import torch
from train_model.helper import run_model, print_result, build_model
from train_model.Engineer import one_stage_eval_model
from config.config_utils import finalize_config
from config.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config yaml file")
    parser.add_argument("--out_prefix", type=str, help="output file name prefix, will append .json or .pkl", default='output')
    parser.add_argument("--log_dir", type=str, help="log directory to store gq", default='boards/default/')
    parser.add_argument("--model_path", type=str, required=True, help="path of model")
    parser.add_argument("--batch_size", type=int, help="batch_size for test", default=None)
    parser.add_argument("--num_workers",type=int, help="num_workers in dataLoader", default=5)
    parser.add_argument("--iter", type=int, help="iteration for storing gq", default=50000)
    parser.add_argument("--json_only", action='store_true', help="flag for only need json result")
    parser.add_argument("--use_val",action='store_true',help="flag for using val data for test")
    parser.add_argument("--store_result",action='store_true',help="flag for storing test results")
    parser.add_argument("--store_questions",action='store_true',help="flag for storing generated questions")
    

    arguments = parser.parse_args()
    return arguments


def multi_gpu_state_to_single(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd


if __name__ == '__main__':

    args = parse_args()

    config_file = args.config
    out_file = args.out_prefix+".json"
    model_file = args.model_path

    finalize_config(cfg, config_file, None)

    batch_size = cfg['data']['batch_size'] if args.batch_size is None else args.batch_size
    if args.use_val:
        data_set_test = prepare_eval_data_set(**cfg['data'], **cfg['model'], verbose=True)
    else:
        data_set_test = prepare_test_data_set(**cfg['data'], **cfg['model'], verbose=True)
    data_reader_test = DataLoader(data_set_test, shuffle=False, batch_size=batch_size, num_workers=args.num_workers)
    ans_dic = data_set_test.answer_dict

    my_model = build_model(cfg, data_set_test)

    sd = torch.load(model_file)['state_dict']

    if list(sd.keys())[0].startswith('module') and not hasattr(my_model, 'module'):
        sd = multi_gpu_state_to_single(sd)

    my_model.load_state_dict(sd)

    my_model.eval()

    print("BEGIN TESTING")
    
    if args.store_result:
        question_ids, soft_max_result = run_model(my_model, data_reader_test, ans_dic.UNK_idx)
        pkl_res_file = args.out_prefix + ".pkl" if not args.json_only else None
        print_result(question_ids, soft_max_result, ans_dic, out_file, args.json_only, pkl_res_file)
        
    else:    
        if args.use_val:
            if args.store_questions:
                acc, ns, _ = one_stage_eval_model(data_reader_test, my_model, i_iter = args.iter, log_dir=args.log_dir)
                print("Validation accuracy : %.6f" % acc)
            else:
                acc, ns, _ = one_stage_eval_model(data_reader_test, my_model)
                print("Validation accuracy : %.6f" % acc)

    print("DONE")


