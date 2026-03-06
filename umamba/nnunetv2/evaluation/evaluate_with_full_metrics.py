import argparse
import multiprocessing
import os
from multiprocessing import Pool
from typing import Tuple, List, Union

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def compute_hausdorff_distance(mask_ref: np.ndarray, mask_pred: np.ndarray, percentile: float = 95.0) -> float:
    if np.sum(mask_ref) == 0 or np.sum(mask_pred) == 0:
        return np.nan
    ref_border = mask_ref ^ binary_erosion(mask_ref)
    pred_border = mask_pred ^ binary_erosion(mask_pred)
    if np.sum(ref_border) == 0:
        ref_border = mask_ref
    if np.sum(pred_border) == 0:
        pred_border = mask_pred
    ref_dist = distance_transform_edt(~ref_border)
    pred_dist = distance_transform_edt(~pred_border)
    ref_to_pred = ref_dist[pred_border]
    pred_to_ref = pred_dist[ref_border]
    all_distances = np.concatenate([ref_to_pred, pred_to_ref])
    if percentile >= 100:
        return np.max(all_distances)
    else:
        return np.percentile(all_distances, percentile)


def compute_assd(mask_ref: np.ndarray, mask_pred: np.ndarray) -> float:
    if np.sum(mask_ref) == 0 or np.sum(mask_pred) == 0:
        return np.nan
    ref_border = mask_ref ^ binary_erosion(mask_ref)
    pred_border = mask_pred ^ binary_erosion(mask_pred)
    if np.sum(ref_border) == 0:
        ref_border = mask_ref
    if np.sum(pred_border) == 0:
        pred_border = mask_pred
    ref_dist = distance_transform_edt(~ref_border)
    pred_dist = distance_transform_edt(~pred_border)
    ref_to_pred = ref_dist[pred_border]
    pred_to_ref = pred_dist[ref_border]
    assd = (np.sum(ref_to_pred) + np.sum(pred_to_ref)) / (len(ref_to_pred) + len(pred_to_ref))
    return assd


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_full_metrics_single_case(args):
    reference_file, prediction_file, image_reader_writer, labels_or_regions, ignore_label, spacing = args
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None
    results = {'reference_file': reference_file, 'prediction_file': prediction_file, 'metrics': {}}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['TP'] = float(tp)
        results['metrics'][r]['FP'] = float(fp)
        results['metrics'][r]['FN'] = float(fn)
        results['metrics'][r]['TN'] = float(tn)
        results['metrics'][r]['Sensitivity'] = tp / (tp + fn) if tp + fn > 0 else np.nan
        results['metrics'][r]['Specificity'] = tn / (tn + fp) if tn + fp > 0 else np.nan
        results['metrics'][r]['Precision'] = tp / (tp + fp) if tp + fp > 0 else np.nan
        if tp + fp > 0 and tp + fn > 0:
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            results['metrics'][r]['F1'] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else np.nan
        else:
            results['metrics'][r]['F1'] = np.nan
        hd95 = compute_hausdorff_distance(mask_ref, mask_pred, percentile=95.0)
        if spacing is not None and not np.isnan(hd95):
            hd95 = hd95 * np.mean(spacing)
        results['metrics'][r]['HD95'] = hd95
        assd = compute_assd(mask_ref, mask_pred)
        if spacing is not None and not np.isnan(assd):
            assd = assd * np.mean(spacing)
        results['metrics'][r]['ASSD'] = assd
    return results


def convert_keys(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def evaluate_with_full_metrics(gt_folder: str, pred_folder: str, dataset_json_file: str, 
                                plans_file: str, output_file: str = None,
                                num_processes: int = default_num_processes):
    print("=" * 60)
    print("Full Metrics Evaluation for Medical Image Segmentation")
    print("=" * 60)
    dataset_json = load_json(dataset_json_file)
    file_ending = dataset_json['file_ending']
    example_file = subfiles(gt_folder, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()
    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    labels_or_regions = lm.foreground_regions if lm.has_regions else lm.foreground_labels
    print(f"\nDataset: {dataset_json.get('name', 'Unknown')}")
    print(f"Labels: {labels_or_regions}")
    print(f"File ending: {file_ending}")
    files_pred = subfiles(pred_folder, suffix=file_ending, join=False)
    files_ref = [join(gt_folder, f) for f in files_pred]
    files_pred_full = [join(pred_folder, f) for f in files_pred]
    print(f"Number of cases: {len(files_pred)}")
    spacing = None
    try:
        _, ref_dict = rw.read_seg(files_ref[0])
        spacing = ref_dict.get('spacing', None)
        if spacing is not None:
            print(f"Image spacing: {spacing}")
    except:
        pass
    print(f"\nComputing metrics using {num_processes} processes...")
    args_list = [(f_ref, f_pred, rw, labels_or_regions, lm.ignore_label, spacing) 
                 for f_ref, f_pred in zip(files_ref, files_pred_full)]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.map(compute_full_metrics_single_case, args_list)
    metric_list = list(results[0]['metrics'][labels_or_regions[0]].keys())
    means = {}
    for r in labels_or_regions:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([res['metrics'][r][m] for res in results])
    foreground_mean = {}
    for m in metric_list:
        values = [means[k][m] for k in means.keys() if k != 0 and k != '0']
        foreground_mean[m] = np.mean(values)
    if output_file is None:
        output_file = join(pred_folder, 'summary_full_metrics.json')
    result = {'metric_per_case': convert_keys(results), 'mean': convert_keys(means), 'foreground_mean': convert_keys(foreground_mean)}
    save_json(result, output_file, sort_keys=True)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nForeground Mean Metrics:")
    print("-" * 40)
    for m in ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'HD95', 'ASSD']:
        if m in foreground_mean:
            suffix = " mm" if m in ['HD95', 'ASSD'] else ""
            print(f"  {m:15s}: {foreground_mean[m]:.4f}{suffix}")
    print("\nPer-Class Mean Metrics:")
    print("-" * 40)
    for r in labels_or_regions:
        print(f"\n  Class {r}:")
        for m in ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'HD95', 'ASSD']:
            if m in means[r]:
                suffix = " mm" if m in ['HD95', 'ASSD'] else ""
                print(f"    {m:15s}: {means[r][m]:.4f}{suffix}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation with full metrics')
    parser.add_argument('--gt_folder', type=str, required=True, help='Ground truth folder')
    parser.add_argument('--pred_folder', type=str, required=True, help='Predictions folder')
    parser.add_argument('--dataset_json', type=str, required=True, help='dataset.json path')
    parser.add_argument('--plans_json', type=str, required=True, help='plans.json path')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file')
    parser.add_argument('-np', '--num_processes', type=int, default=default_num_processes, help='Num processes')
    args = parser.parse_args()
    evaluate_with_full_metrics(args.gt_folder, args.pred_folder, args.dataset_json, args.plans_json, args.output, args.num_processes)


if __name__ == '__main__':
    main()
