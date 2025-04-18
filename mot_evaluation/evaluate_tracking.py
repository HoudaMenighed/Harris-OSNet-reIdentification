import os
import numpy as np
import argparse
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from easydict import EasyDict as edict
from utils.io import read_txt_to_struct, read_seqmaps, \
    extract_valid_gt_data, print_metrics
from utils.bbox import bbox_overlap
from utils.measurements import clear_mot_hungarian, idmeasures


def write_metrics_to_file(output_file, fp, fn, ids, gt ,mota, motp):
    with open(output_file, 'w') as f:
        f.write(f"{fp}\n")
        f.write(f"{fn}\n")
        f.write(f"{ids}\n")
        f.write(f"{gt}\n")
        f.write(f"{mota:.5f}\n")
        f.write(f"{motp:.5f}\n")

def preprocessingDB(trackDB, gtDB, distractor_ids, iou_thres, minvis):
    track_frames = np.unique(trackDB[:, 0])
    gt_frames = np.unique(gtDB[:, 0])
    nframes = min(len(track_frames), len(gt_frames))
    print('n farme : ', nframes)
    res_keep = np.ones((trackDB.shape[0], ), dtype=float)
    for i in range(1, nframes + 1):
        res_in_frame = np.where(trackDB[:, 0] == i)[0]
        res_in_frame_data = trackDB[res_in_frame, :]
        gt_in_frame = np.where(gtDB[:, 0] == i)[0]
        gt_in_frame_data = gtDB[gt_in_frame, :]
        res_num = res_in_frame.shape[0]
        gt_num = gt_in_frame.shape[0]
        overlaps = np.zeros((res_num, gt_num), dtype=float)
        for gid in range(gt_num):
            overlaps[:, gid] = bbox_overlap(
                res_in_frame_data[:, 2:6], gt_in_frame_data[gid, 2:6])
        matched_indices = linear_assignment(1 - overlaps)
        for matched in zip(*matched_indices):
            if overlaps[matched[0], matched[1]] < iou_thres:
                continue
            if gt_in_frame_data[matched[1], 1] in distractor_ids:
                res_keep[res_in_frame[matched[0]]] = 0
            if gt_in_frame_data[matched[1], 8] < minvis:
                res_keep[res_in_frame[matched[0]]] = 0
        frame_id_pairs = res_in_frame_data[:, :2]
        uniq_frame_id_pairs = np.unique(frame_id_pairs)
        has_duplicates = uniq_frame_id_pairs.shape[0] < frame_id_pairs.shape[0]
        assert not has_duplicates, \
            'Duplicate ID in same frame [Frame ID: {}].'.format(i)
    keep_idx = np.where(res_keep == 1)[0]
    print('[TRACK PREPROCESSING]: remove distractors and low visibility boxes,'
          'remaining {}/{} computed boxes'.format(
              len(keep_idx), len(res_keep)))
    trackDB = trackDB[keep_idx, :]
    print('Distractors: {}'.format(', '.join(list(map(str, distractor_ids.astype(int))))))
    keep_idx = np.array([i for i in range(
        gtDB.shape[0]) if gtDB[i, 1] not in distractor_ids and
        gtDB[i, 8] >= minvis])
    print('[GT PREPROCESSING]: Removing distractor boxes, '
          'remaining {}/{} boxes'.format(len(keep_idx), gtDB.shape[0]))
    gtDB = gtDB[keep_idx, :]
    return trackDB, gtDB

def evaluate_sequence(trackDB, gtDB, distractor_ids, iou_thres=0.5, minvis=0):
    trackDB, gtDB = preprocessingDB(
        trackDB, gtDB, distractor_ids, iou_thres, minvis)
    mme, c, fp, g, missed, d, M, allfps = clear_mot_hungarian(
        trackDB, gtDB, iou_thres)
    gt_frames = np.unique(gtDB[:, 0])
    gt_ids = np.unique(gtDB[:, 1])
    st_ids = np.unique(trackDB[:, 1])
    f_gt = len(gt_frames)
    n_gt = len(gt_ids)
    n_st = len(st_ids)
    FN = sum(missed)
    FP = sum(fp)
    IDS = sum(mme)
    MOTP = (sum(sum(d)) / sum(c)) * 100
    MOTAL = (1 - (
        sum(fp) + sum(missed) + np.log10(sum(mme) + 1)) / sum(g)) * 100
    MOTA = (1 - (sum(fp) + sum(missed) + sum(mme)) / sum(g)) * 100
    recall = sum(c) / sum(g) * 100
    precision = sum(c) / (sum(fp) + sum(c)) * 100
    FAR = sum(fp) / f_gt
    MT_stats = np.zeros((n_gt, ), dtype=float)
    for i in range(n_gt):
        gt_in_person = np.where(gtDB[:, 1] == gt_ids[i])[0]
        gt_total_len = len(gt_in_person)
        gt_frames_tmp = gtDB[gt_in_person, 0].astype(int)
        gt_frames_list = list(gt_frames)
        st_total_len = sum(
            [1 if i in M[gt_frames_list.index(f)].keys() else 0
                for f in gt_frames_tmp])
        ratio = float(st_total_len) / gt_total_len
        if ratio < 0.2:
            MT_stats[i] = 1
        elif ratio >= 0.8:
            MT_stats[i] = 3
        else:
            MT_stats[i] = 2
    ML = len(np.where(MT_stats == 1)[0])
    PT = len(np.where(MT_stats == 2)[0])
    MT = len(np.where(MT_stats == 3)[0])
    fr = np.zeros((n_gt, ), dtype=int)
    M_arr = np.zeros((f_gt, n_gt), dtype=int)
    for i in range(f_gt):
        for gid in M[i].keys():
            M_arr[i, gid] = M[i][gid] + 1
    for i in range(n_gt):
        occur = np.where(M_arr[:, i] > 0)[0]
        occur = np.where(np.diff(occur) != 1)[0]
        fr[i] = len(occur)
    FRA = sum(fr)
    idmetrics = idmeasures(gtDB, trackDB, iou_thres)
    metrics = [idmetrics.IDF1, idmetrics.IDP, idmetrics.IDR, recall,
               precision, FAR, n_gt, MT, PT, ML, FP, FN, IDS, FRA,
               MOTA, MOTP, MOTAL]
    extra_info = edict()
    extra_info.mme = sum(mme)
    extra_info.c = sum(c)
    extra_info.fp = sum(fp)
    extra_info.g = sum(g)
    extra_info.missed = sum(missed)
    extra_info.d = d
    extra_info.f_gt = f_gt
    extra_info.n_gt = n_gt
    extra_info.n_st = n_st
    extra_info.ML = ML
    extra_info.PT = PT
    extra_info.MT = MT
    extra_info.FRA = FRA
    extra_info.idmetrics = idmetrics
    return metrics, extra_info

def evaluate_bm(all_metrics):
    f_gt, n_gt, n_st = 0, 0, 0
    nbox_gt, nbox_st = 0, 0
    c, g, fp, missed, ids = 0, 0, 0, 0, 0
    IDTP, IDFP, IDFN = 0, 0, 0
    MT, ML, PT, FRA = 0, 0, 0, 0
    overlap_sum = 0
    for i in range(len(all_metrics)):
        nbox_gt += all_metrics[i].idmetrics.nbox_gt
        nbox_st += all_metrics[i].idmetrics.nbox_st
        IDTP += all_metrics[i].idmetrics.IDTP
        IDFP += all_metrics[i].idmetrics.IDFP
        IDFN += all_metrics[i].idmetrics.IDFN
        MT += all_metrics[i].MT
        ML += all_metrics[i].ML
        PT += all_metrics[i].PT
        FRA += all_metrics[i].FRA
        f_gt += all_metrics[i].f_gt
        n_gt += all_metrics[i].n_gt
        n_st += all_metrics[i].n_st
        c += all_metrics[i].c
        g += all_metrics[i].g
        fp += all_metrics[i].fp
        missed += all_metrics[i].missed
        ids += all_metrics[i].mme
        overlap_sum += sum(sum(all_metrics[i].d))
    IDP = IDTP / (IDTP + IDFP) * 100
    IDR = IDTP / (IDTP + IDFN) * 100
    IDF1 = 2 * IDTP / (nbox_gt + nbox_st) * 100
    FAR = fp / f_gt
    MOTP = (overlap_sum / c) * 100
    MOTAL = (1 - (fp + missed + np.log10(ids + 1)) / g) * 100
    MOTA = (1 - (fp + missed + ids) / g) * 100
    recall = c / g * 100
    precision = c / (fp + c) * 100
    metrics = [IDF1, IDP, IDR, recall, precision, FAR, n_gt,
               MT, PT, ML, fp, missed, ids, FRA, MOTA, MOTP, MOTAL]
    return metrics

def evaluate_tracking(sequences, track_dir, gt_dir, output_file):
    all_info = []
    print('sequences ',sequences)
    for seqname in sequences:
        print('sqname :',seqname)
        track_res = track_dir
        gt_file = gt_dir
        assert os.path.exists(track_res) and os.path.exists(gt_file), \
            'Either tracking result {} or '\
            'groundtruth directory {} does not exist'.format(
                track_res, gt_file)
        trackDB = read_txt_to_struct(track_res)
        gtDB = read_txt_to_struct(gt_file)
        gtDB, distractor_ids = extract_valid_gt_data(gtDB)
        metrics, extra_info = evaluate_sequence(trackDB, gtDB, distractor_ids)
        print_metrics(seqname + ' Evaluation', metrics)
        all_info.append(extra_info)
    all_metrics = evaluate_bm(all_info)
    print_metrics('Summary Evaluation', all_metrics)
    # Write MOTA and MOTP to the specified file
    fp = all_metrics[10]
    fn = all_metrics[11]
    ids = all_metrics[12]
    gt = all_metrics[6]
    mota = all_metrics[14]
    motp = all_metrics[15]

    write_metrics_to_file(output_file,fp, fn, ids, gt, mota, motp)

def parse_args():
    parser = argparse.ArgumentParser(description='MOT Evaluation Toolkit')
    parser.add_argument(
        '--bm', help='Evaluation multiple videos', action='store_true')
    parser.add_argument('--seqmap', help='seqmap file', type=str)
    parser.add_argument('--track', help='Tracking result directory', type=str)
    parser.add_argument(
        '--gt', help='Ground-truth annotation directory', type=str)
    parser.add_argument('--output', help='Output file for MOTA and MOTP', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    sequences = read_seqmaps(args.seqmap)
    evaluate_tracking(sequences, args.track, args.gt, args.output)

