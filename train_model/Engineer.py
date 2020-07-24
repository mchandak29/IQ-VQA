import os
import gc
import sys
import torch
import copy
import shutil
import logging
import json
import torch.nn as nn
import numpy as np
import timeit
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
from global_variables.global_variables import use_cuda
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from config.config import cfg
from tqdm import tqdm


def masked_unk_softmax(x, dim, mask_idx):
    x1 = F.softmax(x, dim=dim)
    x1[:, mask_idx] = 0
    x1_sum = torch.sum(x1, dim=1, keepdim=True)
    y = x1 / x1_sum
    return y

def compute_score_with_logits(logits, labels):
    logits = masked_unk_softmax(logits, 1, 0)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())
    one_hots = one_hots.cuda() if use_cuda else one_hots
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores

def obtain_vocabs(cfg):
    q_vocab_path = os.path.join(cfg.data.data_root_dir, cfg.data.vocab_question_file)
    a_vocab_path = os.path.join(cfg.data.data_root_dir, cfg.data.vocab_answer_file)

    q_vocab = [l.rstrip() for l in tuple(open(q_vocab_path))]
    q_vocab.extend(["<start>", "<end>"])
    a_vocab = [l.rstrip() for l in tuple(open(a_vocab_path))]
    return q_vocab, a_vocab

def one_stage_train(
    myModel,
    data_reader_trn,
    myOptimizer,
    loss_criterion,
    snapshot_dir,
    log_dir,
    i_iter,
    start_epoch,
    data_reader_eval=None,
    scheduler=None,
):

    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter

    is_question_consistency = hasattr(myModel, "question_consistency")

    vocab, ans_vocab = obtain_vocabs(cfg)

    if isinstance(myModel, torch.nn.DataParallel):
        is_question_consistency = hasattr(myModel.module, "question_consistency")

    # Set model to train
    myModel = myModel.train()

    avg_accuracy = 0
    accuracy_decay = 0.99
    best_val_accuracy = 0
    best_val_precision = 0.0
    writer = SummaryWriter(log_dir)
    best_epoch = 0
    best_iter = i_iter
    best_epoch = 0
    iepoch = start_epoch
    start = timeit.default_timer()
    confusion_mat = np.zeros((2, 2))
    val_confusion_mat = np.zeros((2, 2))

    while i_iter < max_iter:
        n_sample_tot = 0

        start_iter = timeit.default_timer()
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break

            answer_scores = batch["ans_scores"]
            answer_scores_cuda = batch["ans_scores"].cuda()
            n_sample = answer_scores.size(0)
            n_sample_tot += n_sample
            myOptimizer.zero_grad()
            
            add_graph = False

            myModel = myModel.train()
            
            if is_question_consistency:
                _return_dict = one_stage_run_model(batch, myModel, add_graph, log_dir)
                logit_res = _return_dict["logits"]
                qc_return_dict = _return_dict["qc_return_dict"]
                qc_loss = qc_return_dict["qc_loss"]
               
                qc_loss = torch.mean(qc_loss, 0)  #calculated only on flagged data
                
            else:
                logit_res = one_stage_run_model(batch, myModel, add_graph, log_dir)[
                    "logits"
                ]

            input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
            if use_cuda:
                input_answers_variable = input_answers_variable.cuda()

            total_loss = loss_criterion(logit_res, input_answers_variable)
            total_loss = total_loss.sum()
            
            if is_question_consistency:
                total_loss += cfg.training_parameters.qc_lambda * qc_loss

            else:
                qc_loss = 0
            
            total_loss.backward()
            
            if max_grad_l2_norm is not None:
                if clip_norm_mode == "all":
                    norm = nn.utils.clip_grad_norm_(
                        myModel.parameters(), max_grad_l2_norm
                    )
                    writer.add_scalar("grad_norm", norm, i_iter)
                elif clip_norm_mode == "question":
                    norm = nn.utils.clip_grad_norm_(
                        myModel.module.question_embedding_models.parameters(),
                        max_grad_l2_norm,
                    )
                    writer.add_scalar("question_grad_norm", norm, i_iter)
                elif clip_norm_mode == "none":
                    pass
                else:
                    raise NotImplementedError

            myOptimizer.step()
            scheduler.step(i_iter)

            if (
                cfg.model.question_consistency.cycle
                and i_iter > cfg["model"]["question_consistency"]["activation_iter"]
            ):
                cycle_batch = {}
                
                for _k, _v in batch.items():
                    cycle_batch[_k] = _v.clone()
                generated_questions = qc_return_dict["sampled_ids"].clone()
                # Preprocess to remove start and end
                generated_questions[generated_questions == len(vocab)-2] = 0
                generated_questions[generated_questions == len(vocab)-1] = 0

                # First letter cannot be unk
                generated_questions = torch.cat(
                    [
                        generated_questions.narrow(
                            1, 1, generated_questions.shape[1] - 1
                        ),
                        generated_questions.narrow(1, 0, 1),
                    ],
                    1,
                )

                # Gating Mechanism
                if cfg["model"]["question_consistency"]["gating_th"] > 0:
                    detached_g_q = generated_questions.clone().detach()
                    detached_g_emb = myModel.question_consistency.embed(
                        detached_g_q
                    ).sum(1)

                    detached_o_q = batch["imp_seq_batch"].long().cuda()
                    detached_o_emb = myModel.question_consistency.embed(
                        detached_o_q
                    ).sum(1)
                    
                    cosine_similarity = F.cosine_similarity(
                        detached_g_emb, detached_o_emb
                    )
                    
                    allowed_indices = (
                        cosine_similarity
                        > cfg["model"]["question_consistency"]["gating_th"]
                    )
#                     print(
#                         "Allowed Batches {}".format(allowed_indices.sum().cpu().item())
#                     )
                else:
                    allowed_indices = torch.ones(len(generated_questions)).long()

                cycle_batch["input_seq_batch"] = generated_questions
                cycle_return_dict = one_stage_run_model(cycle_batch, myModel)
                
                
                ############### Compare with Implied Answer ground truth value #######################
                
                allowed_indices*= batch["flag"].cuda() 
                
                if allowed_indices.sum() > -1:
                    cycle_vqa_loss = cfg.training_parameters.cc_lambda * loss_criterion(
                        cycle_return_dict["logits"][allowed_indices],
                        cycle_batch["imp_ans_scores"][allowed_indices].cuda(),
                    )
                    
                    # perform backward pass
                    cycle_vqa_loss.sum().backward()

                    myOptimizer.step()

            scores = torch.sum(
                compute_score_with_logits(logit_res, input_answers_variable.data)
            )
            accuracy = scores / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            if i_iter % report_interval == 0:
                cur_loss = total_loss.item()
                end_iter = timeit.default_timer()
                time = end_iter - start_iter
                start_iter = timeit.default_timer()
                val_batch = next(iter(data_reader_eval))
                val_score, val_loss = evaluate_a_batch(
                    val_batch, myModel, loss_criterion
                )

                print(
                    "iter:",
                    i_iter,
                    "train_loss: %.4f" % cur_loss,
                    " train_score: %.4f" % accuracy,
                    " qc_loss: %.4f" % qc_loss,
                    " avg_train_score: %.4f" % avg_accuracy,
                    "val_score: %.4f" % val_score,
                    "val_loss: %.4f" % val_loss,
                    "time(s): %.1f" % time,
                )
                sys.stdout.flush()

                writer.add_scalar("train_loss", cur_loss, i_iter)
                writer.add_scalar("train_score", accuracy, i_iter)
                writer.add_scalar("train_score_avg", avg_accuracy, i_iter)
                writer.add_scalar("val_score", val_score, i_iter)
                writer.add_scalar("val_loss", val_loss, i_iter)

            if (i_iter % snapshot_interval == 1 or i_iter == max_iter) and i_iter != 1:
                ##evaluate the model when finishing one epoch
                if data_reader_eval is not None:
                    val_accuracy, upbound_acc, val_sample_tot, val_confusion_mat = one_stage_eval_model(
                        data_reader_eval,
                        myModel,
                        i_iter=i_iter,
                        log_dir=log_dir,
                    )
                    
                    end = timeit.default_timer()
                    epoch_time = end - start
                    start = timeit.default_timer()
                    print(
                        "i_epoch:",
                        iepoch,
                        "i_iter:",
                        i_iter,
                        "val_acc:%.4f" % val_accuracy,
                        "runtime(s):%d" % epoch_time,
                    )
                    sys.stdout.flush()

                model_snapshot_file = os.path.join(
                    snapshot_dir, "model_%08d.pth" % i_iter
                )
                model_result_file = os.path.join(
                    snapshot_dir, "result_%08d.txt" % i_iter
                )
                torch.save(
                    {
                        "epoch": iepoch,
                        "iter": i_iter,
                        "state_dict": myModel.state_dict(),
                        "optimizer": myOptimizer.state_dict(),
                    },
                    model_snapshot_file,
                )
                with open(model_result_file, "w") as fid:
                    fid.write("%d:%.5f\n" % (iepoch, val_accuracy * 100))

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = iepoch
                    best_iter = i_iter
                    best_model_snapshot_file = os.path.join(
                        snapshot_dir, "best_model.pth"
                    )
                    shutil.copy(model_snapshot_file, best_model_snapshot_file)

        gc.collect()

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print(
        "best_acc:%.6f after epoch: %d/%d at iter %d"
        % (best_val_accuracy, best_epoch, iepoch, best_iter)
    )
    sys.stdout.flush()


def evaluate_a_batch(batch, myModel, loss_criterion):
    answer_scores = batch["ans_scores"]
    answer_scores_cuda = answer_scores.cuda()
    n_sample = answer_scores.size(0)

    input_answers_variable = Variable(answer_scores.type(torch.FloatTensor))
    input_answers_variable = (
        input_answers_variable.cuda() if use_cuda else input_answers_variable
    )

    # Set model to eval
    myModel = myModel.eval()

    is_question_consistency = hasattr(myModel, "question_consistency")

    if isinstance(myModel, torch.nn.DataParallel):
        is_question_consistency = (
            True if hasattr(myModel.module, "question_consistency") else False
        )

    if is_question_consistency:
        _return_dict = one_stage_run_model(batch, myModel)
        logit_res = _return_dict["logits"]
        qc_return_dict = _return_dict["qc_return_dict"]
        qc_loss = torch.mean(qc_return_dict["qc_loss"])

    else:
        logit_res = one_stage_run_model(batch, myModel)["logits"]

    predicted_scores = torch.sum(
        compute_score_with_logits(logit_res, input_answers_variable.data)
    )

    total_loss = loss_criterion(logit_res, input_answers_variable)


    if is_question_consistency:
        total_loss += qc_loss


    gc.collect()
    return predicted_scores / n_sample, total_loss.item()


def one_stage_eval_model(
    data_reader_eval,
    myModel,
    thresholding_type=None,
    threshold=0.5,
    return_cm=False,
    i_iter=None,
    log_dir=None,
):

    val_score_tot = 0
    val_sample_tot = 0
    upbound_tot = 0
    writer = SummaryWriter(log_dir)

    # Set model to eval
    myModel = myModel.eval()
    vocab, ans_vocab = obtain_vocabs(cfg)

    # Make dict to store generated questions
    gq_dict = {"annotations": [], "ques_answers": []}

    def store_questions(att, batch, sampled_ids = None):
        
        images = batch["image_id"]
        att = att.data.cpu().numpy()
        
        orig_q = batch["input_seq_batch"].data.cpu().numpy()
        orig_questions = [[vocab[idx] for idx in orig_q[j]] for j in range(len(orig_q))]
        
        orig_answers = batch["answer_label_batch"].data.cpu().numpy()
        verbose_info = batch['verbose_info']
        q_ids = verbose_info['question_id'].cpu().numpy()
        
        for jdx, (oq, img, oa,qid, at) in enumerate(zip( orig_questions, images, orig_answers,q_ids, att)):
            gq_dict["ques_answers"] += [{"image_id": int(img), "orig_ques": " ".join(oq), "orig_ans": ans_vocab[oa],
                                         "attention": at, "ques_id": int(qid)}]

        
        if sampled_ids is not None:
            sampled_ids = sampled_ids.data.cpu().numpy()
            questions = [[vocab[idx] for idx in sampled_ids[j]] for j in range(len(sampled_ids))]
            gt_q = batch["imp_seq_batch"].data.cpu().numpy()
            gt_questions = [[vocab[idx] for idx in gt_q[j]] for j in range(len(gt_q))]
        
            imp_answers = batch["imp_answer_label_batch"].data.cpu().numpy()
       
            for jdx, (q, gtq,img,ia, qid) in enumerate(zip(questions, gt_questions, images, imp_answers, q_ids)):
                gq_dict["annotations"] += [{"image_id": int(img), "imp_ans": ans_vocab[ia], "gen_ques": " ".join(q),
                                            "gt_gen_ques": " ".join(gtq), "ques_id": int(qid)}]
                    

    is_question_consistency = (
        True if hasattr(myModel, "question_consistency") else False
    )

    if isinstance(myModel, torch.nn.DataParallel):
        is_question_consistency = (
            True if hasattr(myModel.module, "question_consistency") else False
        )

    for idx, batch in tqdm(enumerate(data_reader_eval)):
        answer_scores = batch["ans_scores"]
        n_sample = answer_scores.size(0)
        answer_scores = answer_scores.cuda() if use_cuda else answer_scores

        if is_question_consistency:
            _return_dict = one_stage_run_model(batch, myModel)
            logit_res = _return_dict["logits"]
            qc_return_dict = _return_dict["qc_return_dict"]
            att = _return_dict['attention']
            if "sampled_ids" in qc_return_dict.keys():
                sampled_ids = qc_return_dict["sampled_ids"]
                store_questions(att, batch, sampled_ids)

        else:
            _return_dict = one_stage_run_model(batch, myModel)
            logit_res = _return_dict["logits"]
            att = _return_dict['attention']
            store_questions(att, batch)
                
        predicted_scores = torch.sum(
            compute_score_with_logits(logit_res, answer_scores)
        )
        upbound = torch.sum(torch.max(answer_scores, dim=1)[0])

        val_score_tot += predicted_scores
        val_sample_tot += n_sample
        upbound_tot += upbound

    gc.collect()

    if log_dir is not None:
        np.save(os.path.join(log_dir, "gq_{}.npy".format(i_iter)), np.array(gq_dict))

    return val_score_tot / val_sample_tot, upbound_tot / val_sample_tot, val_sample_tot

def one_stage_run_model(batch, myModel, add_graph=False, log_dir=None,
                        normalize=False):
    input_text_seqs = batch["input_seq_batch"]
    input_images = batch["image_feat_batch"]
    input_txt_variable = Variable(input_text_seqs.type(torch.LongTensor))
    input_txt_variable = input_txt_variable.cuda() if use_cuda else input_txt_variable
    
    # Load Implied Question Ground Truth
    imp_seqs = batch["imp_seq_batch"]
    imp_seq_variable = Variable(imp_seqs.type(torch.LongTensor))
    imp_seq_variable = imp_seq_variable.cuda() if use_cuda else imp_seq_variable
    
    # Load Implied Answer Ground Truth
    imp_ans_gt = batch["imp_ans_scores"]
    imp_ans_variable = Variable(imp_ans_gt)
    imp_ans_variable = imp_ans_variable.cuda() if use_cuda else imp_ans_variable
    
    # Load Original Answer Ground Truth
    ans_gt = batch["ans_scores"]
    ans_variable = Variable(ans_gt)
    ans_variable = ans_variable.cuda() if use_cuda else ans_variable
    
    ############## Load Implication Flag ##################
    imp_flag = batch["flag"]
    imp_flag_var = Variable(imp_flag)
    imp_flag_var = imp_flag_var.cuda() if use_cuda else imp_flag_var
    
    ########## Load implication type ###########
    imp_type = batch["imp_type"]
    imp_type_var = Variable(imp_type)
    imp_type_var = imp_type_var.cuda() if use_cuda else imp_type_var

    if isinstance(input_images, list):
        input_images = input_images[0]

    image_feat_variable = Variable(input_images)
    image_feat_variable = (
        image_feat_variable.cuda() if use_cuda else image_feat_variable
    )
    image_feat_variables = [image_feat_variable]

    image_dim_variable = None
    if "image_dim" in batch:
        image_dims = batch["image_dim"]
        image_dim_variable = Variable(image_dims, requires_grad=False, volatile=False)
        image_dim_variable = (
            image_dim_variable.cuda() if use_cuda else image_dim_variable
        )

    # check if more than 1 image_feat_batch
    i = 1
    image_feat_key = "image_feat_batch_%s"
    while image_feat_key % str(i) in batch:
        tmp_image_variable = Variable(batch[image_feat_key % str(i)])
        tmp_image_variable = (
            tmp_image_variable.cuda() if use_cuda else tmp_image_variable
        )
        image_feat_variables.append(tmp_image_variable)
        i += 1

    return_dict = myModel(
        input_question_variable=input_txt_variable,
        image_dim_variable=image_dim_variable,
        image_feat_variables=image_feat_variables,
        imp_gt_ques = imp_seq_variable,
        imp_gt_ans = imp_ans_variable,
        imp_flag = imp_flag_var,
        gt_ans = ans_variable,
        imp_type = imp_type_var,
        batch=batch,
    )

    if add_graph:
        with SummaryWriter(log_dir=log_dir, comment="basicblock") as w:
            w.add_graph(
                myModel, (input_txt_variable, image_dim_variable, image_feat_variables)
            )

    if normalize:
        return_dict["logits"] = masked_unk_softmax(return_dict["logits"], 1, 0)

    return return_dict