import numpy
import pickle
import json
import os
import argparse


def print_consistency():
	
	val_imdb = numpy.load('data/imdb/imdb_val2014.npy',allow_pickle=True)
	manual_imdb = numpy.load('data/imdb_manual/imdb_val2014.npy',allow_pickle=True)

	val_pred = json.load(open('results/pythia_cycle_consistent/3014/' + 'Pythia_IC_val.json','r'))
	manual_pred = json.load(open('results/pythia_cycle_consistent/3014/' + 'Pythia_IC_man_val.json','r'))

	val_map = {}
	manual_map = {}

	for val in manual_pred:
	    manual_map[val['question_id']] = val['answer']
	    
	for val in val_pred:
	    val_map[val['question_id']] = val['answer']


	for i in val_imdb[1:]:
	    if val_map[i['question_id']] not in i['valid_answers']:
	        val_map.pop(i['question_id'])

	count_correct = [0,0,0]
	count_total = [0,0,0]

	for i in manual_imdb[1:]:
	    qid = int(i['question_id']/10)
	    if qid in val_map:
	        _type = i['question_id']%10 - 1
	        count_total[_type]+=1
	        if manual_map[i['question_id']] in i['valid_answers']:
	            count_correct[_type]+=1

	for i,_type in enumerate(['Logeq','Necc','Mutex']):
	    print(_type + ': %0.4f %d' % (count_correct[i]/count_total[i], count_total[i]))
	print('ALL : ',sum(count_correct)/sum(count_total))


if __name__ == '__main__':
	print_consistency()