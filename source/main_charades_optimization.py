#!/usr/bin/python
import argparse
import os
import time
from svc_uap import svc_rp
from proposals_utils import init_log, frame_to_time, fill_log_file, write_res
from data_utils import get_videos, get_vid_info
from evaluation_utils import run_evaluation, plot_metric
from tqdm import tqdm
import json

def parse_input_arguments():

    description = 'SVC-UAP. Unsupervised Temporal Action Proposals.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-data', choices=['ActivityNet', 'Thumos14', 'Charades'],
                   default='ActivityNet', help='Dataset')
    p.add_argument('-gt', default='../gt/activity_net.v1-3.min.json',
                   help='Ground truth file in .json format.')
    p.add_argument('-h5', default='../h5/c3d-activitynet.hdf5',
                   help='Features in .hdf5 file.')
    p.add_argument('-set', default='training',
                   choices=['training', 'validation', 'testing', 'Test'],
                   help='Dataset subset.')
    p.add_argument('-init_n', type=int, default=256, # 
                   help='Initial number of samples to take when starting the '
                        'algorithm or a new proposal.')
    p.add_argument('-n', type=int, default=256, # 
                   help='Number of new samples to take when analysing the same'
                        'proposal.')
    p.add_argument('-th', type=float, default=1, # 0.2
                   help='Classification error rate threshold.')
    p.add_argument('-c', type=float, default= 0.019306, # 0.019306
                   help='C parameter of the of the Linear SVM.')
    p.add_argument('-rpth', type=float, default=1,
                   help='Rank-pooling threshold.') # 0.8
    p.add_argument('-res', default='../res/svc-uap-res.json',
                   help='Proposals result.')
    p.add_argument('-eval', action='store_false',
                   help='Use this variable if only evaluation is needed.')
    p.add_argument('-log', default='../log/svc-uap.log',
                   help='Log file with execution information.')
    p.add_argument('-fig', default='../res/ar-an.png',
                   help='Figure with AR-AN metric.')
    p.add_argument('-part', type=int, default=1)

    return p.parse_args()


def svc_uap(data, vid_names, gt, h5, init_n, n, th, c, log, rp_th):

    ppsals_data = {'results': {},
                   'version': 'VERSION 1.3',
                   'external_data': {},
                   }

    for idx, vid in enumerate(vid_names):
        print(f'Processing video {idx} / {len(vid_names)} ...')

        d, fps = get_vid_info(vid, gt, h5, data)

        ppsals, score = svc_rp(d, init_n, n, c, th, rp_th)

        temp_ppsals = frame_to_time(ppsals, fps, data)

        this_vid_ppsals = []
        for i in range(0, temp_ppsals.shape[0]):
            ppsal = {'score': score[i],
                     'segment': [temp_ppsals[i, 0], temp_ppsals[i, 1]],
                     }
            this_vid_ppsals += [ppsal]

        ppsals_data['results'][vid] = this_vid_ppsals
        if idx == len(vid_names) - 1:
            fill_log_file(idx+1, vid, temp_ppsals.shape[0], 1, log)
        else:
            fill_log_file(idx, vid, temp_ppsals.shape[0], 0, log)

    return ppsals_data


def main(data, gt, h5, set, init_n, n, th, c, rpth, res, eval, log, fig, part):

    if eval:
        if not os.path.exists('../log'):
            os.makedirs('../log')
        init_log(set, init_n, n, th, c, rpth, log)
        vid_names = get_videos(gt, set)

        print('Running svm clustering.')
        start_time = time.time()
        ppsals = svc_uap(data, vid_names, gt, h5, init_n, n, th, c, log, rpth)
        elapsed_time = time.time() - start_time
        print('Execution time in seconds = ' + str(elapsed_time))

        if not os.path.exists('../res'):
            os.makedirs('../res')
        write_res(ppsals, res)

    print('Running evaluation.')
    if not os.path.isfile(res):
        print(f'{res}: No such file or directory.')
        return
    else:
        average_n_proposals, average_recall, recall, ar_an, num_p = run_evaluation(gt, res,
                                                                 subset=set)
        plot_metric(average_n_proposals, average_recall, recall, fig_file=fig)

    return ar_an, num_p


if __name__ == '__main__':
    args = parse_input_arguments()
    argc_origin = args.c
    args.log = '../log/charades_optimizing'
    train_gt = '/workspace/gt/charades_sta_train_origin.txt'
    tIoU_save_dir = '/workspace/save_charades_optimization_result'
    with open(train_gt, 'r') as f:
        tmpgt = f.readlines()
    
    charades_train_keys = set([k.split('##')[0].split(' ')[0].strip() for k in tmpgt])
    
    existing = set()
    new_dict = {}
    for line_ in sorted(tmpgt):
        vid = str(line_.split('##')[0].split(' ')[0].strip())
        st = float(line_.split('##')[0].split(' ')[1].strip())
        end = float(line_.split('##')[0].split(' ')[2].strip())
        seg = [st, end]
        
        if vid in existing:
            old_seg = new_dict[vid]
            old_seg.append(seg)
            new_dict[vid] = old_seg
        else :
            new_dict[vid] = [seg]
        
        existing.add(vid)
    
    if args.part == 1:
        args.log = args.log + '_part_1'
        tIoU_save_dir = tIoU_save_dir + '_part_1.txt'
        for i_ in range(1, 25, 5):
            for j_ in range(1, 25, 5):
                for k_ in range(0, 11, 2):
                    i = i_ / 100
                    j = j_ / 100
                    args.rpth = i
                    args.th = j
                    args.c = argc_origin / pow(10, k_)
                    args.res = '../res/' + args.data + '_' + args.set + '_c_' + str(args.c) + '_result_rpth_' + str(args.rpth) + '_th_' + str(args.th) + '.json'
                    ar_an, num_p = main(**vars(args))

                    rank1_miou = []
                    gt_miou = []
                    
                    with open(args.res, 'r') as f:
                        tmpnew = json.load(f)['results']

                    for key in tqdm(charades_train_keys):
                        if key not in tmpnew.keys():
                            continue
                        segments = tmpnew[key]
                        # duration = tmpnew[key]['duration']
                        segs_gt = new_dict[key]
                        max_iou = 0
                        for seg in segs_gt:
                            for segment in segments:
                            # print(seg, segment)
                                # print(type(segment[1]))
                                segment = segment['segment']
                                all_len = max(seg[1], segment[1]) - min(seg[0], segment[0])
                                joint_len = min(seg[1], segment[1]) - max(seg[0], segment[0])
                                
                                iou = joint_len / all_len
                                if joint_len < 0:
                                    iou = 0
                                if iou > max_iou:
                                    max_iou = iou
                            gt_miou.append(max_iou)
                        rank1_miou.append(max_iou)

                    iou = sum(rank1_miou) / len(rank1_miou)
                    gtiou = sum(gt_miou) / len(gt_miou)

                    with open(tIoU_save_dir, 'a') as f:
                        f.writelines(args.res + '\trank1 mIoU : ' + str(iou) + '\tdirect comp with GT : ' + str(gtiou) + '\tarea under ar vs an: ' + str(ar_an) + '\tnum of proposals: ' + str(num_p) + '\n')
                            # print(max_iou)
                    # calculate tIoU
    elif args.part == 2:
        args.log = args.log + '_part_2'
        tIoU_save_dir = tIoU_save_dir + '_part_2.txt'
        for i_ in range(25, 50, 5):
            for j_ in range(25, 50, 5):
                for k_ in range(0, 11, 2):
                    i = i_ / 100
                    j = j_ / 100
                    args.rpth = i
                    args.th = j
                    args.c = argc_origin / pow(10, k_)
                    args.res = '../res/' + args.data + '_' + args.set + '_c_' + str(args.c) + '_result_rpth_' + str(args.rpth) + '_th_' + str(args.th) + '.json'
                    ar_an, num_p = main(**vars(args))

                    rank1_miou = []
                    gt_miou = []
                    
                    with open(args.res, 'r') as f:
                        tmpnew = json.load(f)['results']

                    for key in tqdm(charades_train_keys):
                        if key not in tmpnew.keys():
                            continue
                        segments = tmpnew[key]
                        # duration = tmpnew[key]['duration']
                        segs_gt = new_dict[key]
                        max_iou = 0
                        for seg in segs_gt:
                            for segment in segments:
                            # print(seg, segment)
                                # print(type(segment[1]))
                                segment = segment['segment']
                                all_len = max(seg[1], segment[1]) - min(seg[0], segment[0])
                                joint_len = min(seg[1], segment[1]) - max(seg[0], segment[0])
                                
                                iou = joint_len / all_len
                                if joint_len < 0:
                                    iou = 0
                                if iou > max_iou:
                                    max_iou = iou
                            gt_miou.append(max_iou)
                        rank1_miou.append(max_iou)

                    iou = sum(rank1_miou) / len(rank1_miou)
                    gtiou = sum(gt_miou) / len(gt_miou)

                    with open(tIoU_save_dir, 'a') as f:
                        f.writelines(args.res + '\trank1 mIoU : ' + str(iou) + '\tdirect comp with GT : ' + str(gtiou) + '\tarea under ar vs an: ' + str(ar_an) + '\tnum of proposals: ' + str(num_p) + '\n')
                            # print(max_iou)
                    # calculate tIoU
    elif args.part == 3:
        args.log = args.log + '_part_3'
        tIoU_save_dir = tIoU_save_dir + '_part_3.txt'
        for i_ in range(50, 75, 5):
            for j_ in range(50, 75, 5):
                for k_ in range(0, 11, 2):
                    i = i_ / 100
                    j = j_ / 100
                    args.rpth = i
                    args.th = j
                    args.c = argc_origin / pow(10, k_)
                    args.res = '../res/' + args.data + '_' + args.set + '_c_' + str(args.c) + '_result_rpth_' + str(args.rpth) + '_th_' + str(args.th) + '.json'
                    ar_an, num_p = main(**vars(args))

                    rank1_miou = []
                    gt_miou = []
                    
                    with open(args.res, 'r') as f:
                        tmpnew = json.load(f)['results']

                    for key in tqdm(charades_train_keys):
                        if key not in tmpnew.keys():
                            continue
                        segments = tmpnew[key]
                        # duration = tmpnew[key]['duration']
                        segs_gt = new_dict[key]
                        max_iou = 0
                        for seg in segs_gt:
                            for segment in segments:
                            # print(seg, segment)
                                # print(type(segment[1]))
                                segment = segment['segment']
                                all_len = max(seg[1], segment[1]) - min(seg[0], segment[0])
                                joint_len = min(seg[1], segment[1]) - max(seg[0], segment[0])
                                
                                iou = joint_len / all_len
                                if joint_len < 0:
                                    iou = 0
                                if iou > max_iou:
                                    max_iou = iou
                            gt_miou.append(max_iou)
                        rank1_miou.append(max_iou)

                    iou = sum(rank1_miou) / len(rank1_miou)
                    gtiou = sum(gt_miou) / len(gt_miou)

                    with open(tIoU_save_dir, 'a') as f:
                        f.writelines(args.res + '\trank1 mIoU : ' + str(iou) + '\tdirect comp with GT : ' + str(gtiou) + '\tarea under ar vs an: ' + str(ar_an) + '\tnum of proposals: ' + str(num_p) + '\n')
                            # print(max_iou)
                    # calculate tIoU
    elif args.part == 4:
        args.log = args.log + '_part_4'
        tIoU_save_dir = tIoU_save_dir + '_part_4.txt'
        for i_ in range(75, 101, 5):
            for j_ in range(75, 101, 5):
                for k_ in range(0, 11, 2):
                    i = i_ / 100
                    j = j_ / 100
                    args.rpth = i
                    args.th = j
                    args.c = argc_origin / pow(10, k_)
                    args.res = '../res/' + args.data + '_' + args.set + '_c_' + str(args.c) + '_result_rpth_' + str(args.rpth) + '_th_' + str(args.th) + '.json'
                    ar_an, num_p = main(**vars(args))

                    rank1_miou = []
                    gt_miou = []
                    
                    with open(args.res, 'r') as f:
                        tmpnew = json.load(f)['results']

                    for key in tqdm(charades_train_keys):
                        if key not in tmpnew.keys():
                            continue
                        segments = tmpnew[key]
                        # duration = tmpnew[key]['duration']
                        segs_gt = new_dict[key]
                        max_iou = 0
                        for seg in segs_gt:
                            for segment in segments:
                            # print(seg, segment)
                                # print(type(segment[1]))
                                segment = segment['segment']
                                all_len = max(seg[1], segment[1]) - min(seg[0], segment[0])
                                joint_len = min(seg[1], segment[1]) - max(seg[0], segment[0])
                                
                                iou = joint_len / all_len
                                if joint_len < 0:
                                    iou = 0
                                if iou > max_iou:
                                    max_iou = iou
                            gt_miou.append(max_iou)
                        rank1_miou.append(max_iou)

                    iou = sum(rank1_miou) / len(rank1_miou)
                    gtiou = sum(gt_miou) / len(gt_miou)

                    with open(tIoU_save_dir, 'a') as f:
                        f.writelines(args.res + '\trank1 mIoU : ' + str(iou) + '\tdirect comp with GT : ' + str(gtiou) + '\tarea under ar vs an: ' + str(ar_an) + '\tnum of proposals: ' + str(num_p) + '\n')
                    # calculate tIoU
