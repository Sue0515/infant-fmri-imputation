"""Export a compact distribution comparison from audit_masks output."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(path, output):
    data = json.loads(Path(path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    real = data['eligible_by_split']['train']['recoverable_motion_known_gaze']
    comparison = data['sampler_comparison']
    summaries = [real, comparison['legacy_mixture']['realized_targets'], comparison['fitted']['realized_targets']]
    labels = ['Real motion targets', 'Previous mixed masks', 'Fitted masks']
    colors = ['#263645', '#db8040', '#21877f']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout='constrained')
    bins = [(1,1), (2,2), (3,4), (5,8), (9,16), (17,float('inf'))]
    for i, (summary, label, color) in enumerate(zip(summaries, labels, colors)):
        hist = {int(k): v for k, v in summary['gap_frames_histogram'].items()}
        heights = [100*sum(v for k, v in hist.items() if lo <= k <= hi)/summary['gaps'] for lo, hi in bins]
        axes[0].bar(np.arange(len(bins))+(i-1)*.25, heights, width=.24, label=label, color=color)
    axes[0].set(xticks=np.arange(len(bins)), xticklabels=['1', '2', '3–4', '5–8', '9–16', '>16'],
                xlabel='Contiguous target length (TR; 1 TR = 2 s)', ylabel='Share of gaps (%)', title='Gap-length distribution')
    axes[0].legend(frameon=False, fontsize=8)
    for axis, key, title, ylabel in [
        (axes[1], 'gap_seconds_quantiles', 'Median / 95th percentile', 'Gap duration (s)'),
        (axes[2], None, 'Acquisition-boundary gaps', 'Share of gaps (%)')]:
        for i, (summary, color) in enumerate(zip(summaries, colors)):
            if key:
                median, high = summary[key]['0.5'], summary[key]['0.95']
                axis.bar(i-.14, median, width=.27, color=color, alpha=.45)
                axis.bar(i+.14, high, width=.27, color=color)
                axis.text(i-.14, median+.25, f'{median:g}', ha='center', fontsize=8)
                axis.text(i+.14, high+.25, f'{high:g}', ha='center', fontsize=8)
            else:
                value = 100*summary['boundary_gap_fraction']
                axis.bar(i, value, color=color, width=.6)
                axis.text(i, value+.1, f'{value:.1f}%', ha='center', fontsize=8)
        axis.set(xticks=np.arange(3), xticklabels=['Real', 'Previous', 'Fitted'], title=title, ylabel=ylabel)
        axis.margins(y=.18)
    fig.suptitle('Infant motion imputation: training-family mask calibration', fontsize=14)
    protocol = data['protocol']
    fig.supxlabel(f"{real['runs']} eligible training runs; {protocol['draws_per_training_record']} draws/run/method. "
                  f"Gaze shift = {protocol['eye_shift_tr']} TR, guard = {protocol['guard_tr']}.\n"
                  'Real: recoverable motion gaps. Simulated: known targets after intersection with valid data.\n'
                  'Fitted donors are family-balanced and condition on nonempty gaps; not exact distribution matching.', fontsize=8)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('mask_comparison.'+extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot(args.audit, args.out)


if __name__ == '__main__':
    main()
