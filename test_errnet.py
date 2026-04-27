import argparse
import json
import os
import sys
from os.path import join

import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options import TrainOptions
from models.errnet_model import ERRNetModel
from util.eval_diagnostics import run_eval_diagnostics
from util.input_channel_weights import first_conv_group_mean_abs_weight


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "testdata_CEILNET_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "postcard",
        "save_subdir": "postcard",
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "objects",
        "save_subdir": "objects",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "wild",
        "save_subdir": "wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
}

TEST_DATASETS = {
    "internet": {
        "path": None,
        "save_subdir": "internet",
    },
}


def parse_test_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(list(EVAL_DATASETS.keys()) + list(TEST_DATASETS.keys()) + ["custom"]),
        help="dataset to run",
    )
    parser.add_argument(
        "--data_root",
        default="./datasets/processed_data",
        help="root directory for processed datasets",
    )
    parser.add_argument(
        "--input_dir",
        default="./datasets/raw_data/CEILNet/testdata_reflection_real",
        help="input directory for test-only datasets such as internet/custom images",
    )
    parser.add_argument(
        "--result_dir",
        default="./results",
        help="directory for saved outputs",
    )
    parser.add_argument(
        "--save_subdir",
        default=None,
        help="override output subdirectory name under result_dir",
    )
    parser.add_argument(
        "--file_tag",
        default=None,
        help="if set, save outputs as errnet_model_<tag>.png, t_label_<tag>.png, m_input_<tag>.png (eval only)",
    )
    parser.add_argument(
        "--metrics_out",
        default=None,
        help="output JSON filename (under result_dir/save_subdir) or absolute path. Default: metrics.json. Use --no_metrics_file to skip files.",
    )
    parser.add_argument(
        "--no_metrics_file",
        action="store_true",
        help="do not write metrics JSON/txt (only print to console).",
    )
    parser.add_argument(
        "--skip_eval_diagnostics",
        action="store_true",
        help="skip 2nd pass: saliency / conv1 activation / cross-image cov (eval only, slower)",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=None,
        help="resize test images so the longest edge does not exceed this value",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _first_layer_group_weights(model, opt):
    if not hasattr(model, "net_i"):
        return {}
    return first_conv_group_mean_abs_weight(model.net_i, opt)


def _print_avg_meters_pretty(avg_meters):
    """One metric per line so terminal does not wrap/truncate a single long string."""
    keys = sorted(avg_meters.keys())
    if not keys:
        return
    for k in keys:
        print("  %s: %.6f" % (k, avg_meters[k]))


def _resolve_metrics_path(cli_args, save_subdir, filename):
    if os.path.isabs(filename):
        mpath = filename
    else:
        mpath = join(cli_args.result_dir, save_subdir, filename)
    parent = os.path.dirname(mpath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return mpath


def _write_metrics_txt(
    mpath_json,
    res_avg_meters,
    wstat,
    diag,
    dataset_name,
):
    """Plain-text summary next to JSON: same base name, .txt extension."""
    mpath = os.path.splitext(mpath_json)[0] + ".txt"
    lines = [
        "dataset: %s" % dataset_name,
        "",
        "== Average image quality ==",
    ]
    if res_avg_meters is not None and len(res_avg_meters.keys()) > 0:
        for k in sorted(res_avg_meters.keys()):
            lines.append("  %s: %.6f" % (k, res_avg_meters[k]))
    else:
        lines.append("  (not computed; no ground truth in this run)")
    if wstat:
        lines.extend(["", "== Conv1 group mean |W| =="])
        for k, v in sorted(wstat.items()):
            lines.append("  %s: %s" % (k, v))
    if diag:
        lines.extend(["", "== Eval diagnostics =="])
        for k, v in sorted(diag.items()):
            lines.append("  %s: %s" % (k, v))
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_eval_dataloader(opt, data_root, dataset_key, max_long_edge=None):
    spec = EVAL_DATASETS[dataset_key]
    dataset = datasets.CEILTestDataset(
        join(data_root, spec["path"]),
        max_long_edge=max_long_edge if max_long_edge is not None else spec.get("max_long_edge"),
    )
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )
    return spec, dataloader


def build_test_dataloader(opt, dataset_key, input_dir, max_long_edge=None):
    if dataset_key == "custom":
        dataset = datasets.RealDataset(input_dir, max_long_edge=max_long_edge)
        save_subdir = "custom"
    else:
        spec = TEST_DATASETS[dataset_key]
        dataset = datasets.RealDataset(
            input_dir if spec["path"] is None else join(input_dir, spec["path"]),
            max_long_edge=max_long_edge,
        )
        save_subdir = spec["save_subdir"]

    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )
    return save_subdir, dataloader


def main():
    cli_args = parse_test_args()
    option_parser = TrainOptions()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False

    engine = Engine(opt)

    if cli_args.dataset in EVAL_DATASETS:
        spec, dataloader = build_eval_dataloader(
            opt,
            cli_args.data_root,
            cli_args.dataset,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or spec["save_subdir"]
        ev_kwargs = {}
        if cli_args.file_tag is not None:
            ev_kwargs["suffix"] = cli_args.file_tag
        res = engine.eval(
            dataloader,
            dataset_name=spec["dataset_name"],
            savedir=join(cli_args.result_dir, save_subdir),
            **ev_kwargs,
        )
        print("Average image quality (per-line):", flush=True)
        _print_avg_meters_pretty(res)
        wstat = _first_layer_group_weights(engine.model, opt)
        print(
            "Conv1 group mean |W| (first layer, by input type):",
            json.dumps(wstat, indent=2, allow_nan=False),
            flush=True,
        )
        diag = {}
        if not cli_args.skip_eval_diagnostics and isinstance(engine.model, ERRNetModel):
            print("Running eval diagnostics (saliency / conv1 activation / feature cov) — 2nd pass over val set…")
            diag = run_eval_diagnostics(engine.model, dataloader)
            print("Saliency |dL/dx| L1 (by input group) / null if branch unused:", json.dumps({k: diag.get(k) for k in (
                "sal_grad_l1_RGB", "sal_grad_l1_DCP", "sal_grad_l1_hyper")}, indent=2, allow_nan=False), flush=True)
            print("Activation @ conv1 (mean_c spatial-mean|F| and mean ||F||_2) / conv1 pooled feature cov (N images):", json.dumps({k: diag.get(k) for k in (
                "act_conv1_mean_l1", "act_conv1_fro_l2",
                "cov_pooled_tr", "cov_pooled_fro", "cov_pooled_off_fro",
            )}, indent=2, allow_nan=False), flush=True)
        if not cli_args.no_metrics_file:
            out_name = cli_args.metrics_out or "metrics.json"
            mpath = _resolve_metrics_path(cli_args, save_subdir, out_name)
            d = {k: float(res[k]) for k in res.keys()}
            d.update(wstat)
            d.update(diag)
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, allow_nan=True)
            _write_metrics_txt(
                mpath, res, wstat, diag, spec["dataset_name"],
            )
            print("Wrote metrics: %s  and  %s" % (mpath, os.path.splitext(mpath)[0] + ".txt"), flush=True)
    else:
        default_save_subdir, dataloader = build_test_dataloader(
            opt,
            cli_args.dataset,
            cli_args.input_dir,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or default_save_subdir
        engine.test(
            dataloader,
            savedir=join(cli_args.result_dir, save_subdir),
        )
        wstat = _first_layer_group_weights(engine.model, opt)
        print("Conv1 group mean |W| (RGB / DCP / hyper):", json.dumps(wstat, indent=2), flush=True)
        if not cli_args.no_metrics_file:
            out_name = cli_args.metrics_out or "metrics.json"
            mpath = _resolve_metrics_path(cli_args, save_subdir, out_name)
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(wstat, f, indent=2, allow_nan=True)
            _write_metrics_txt(
                mpath, None, wstat, {}, str(cli_args.dataset),
            )
            print("Wrote metrics: %s  and  %s" % (mpath, os.path.splitext(mpath)[0] + ".txt"), flush=True)


if __name__ == "__main__":
    main()

#图像中给了原图就计算生成图和原图之间的差异，否则就仅生成图像