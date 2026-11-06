"""Plot saved scan trajectories without Isaac Gym or model checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.load(args.run_dir/'trajectory.npz', allow_pickle=False)
    report = json.loads((args.run_dir/'report.json').read_text())
    time = np.arange(len(data['q']))*report['control_dt']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(data['axis_angle'].shape[1]):
        axes[0, 0].plot(time, np.rad2deg(data['axis_angle'][:, i]), label='env %d' % i)
    axes[0, 0].axhline(360, color='gray', linestyle='--', linewidth=.8)
    axes[0, 0].set(title='Measured object rotation (simulation truth)', xlabel='Time (s)', ylabel='Net rotation (deg)')
    axes[0, 0].legend()
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
    names = ['index', 'middle', 'ring', 'thumb']
    for i, name in enumerate(names):
        axes[0, 1].plot(time, data['object_gel_load'][:, 0, i], color=colors[i], linewidth=.7, label=name)
    axes[0, 1].set(title='Object-specific gel load, env 0 (evaluation)', xlabel='Time (s)', ylabel='Normal load (N)')
    axes[0, 1].legend()
    phases = data['phase_names'].tolist() if 'phase_names' in data else ['settle', 'approach', 'hold', 'rotate', 'release', 'reset', 'fault']
    axes[1, 0].step(time, data['phase'][:, 0], where='post', linewidth=.8)
    axes[1, 0].set(title='Controller phases, env 0', xlabel='Time (s)', yticks=range(len(phases)), yticklabels=phases)
    points = data['contact_samples']
    for i, name in enumerate(names):
        selected = points[(points[:, 1] == 0) & (points[:, 2] == i)]
        selected = selected[::max(1, len(selected)//1500)]
        axes[1, 1].scatter(selected[:, 3]*1000, selected[:, 4]*1000, s=2, alpha=.4, color=colors[i], label=name)
    axes[1, 1].set(title='Contact locations in object frame, env 0\nSimulator samples, not a tactile reconstruction',
                   xlabel='Object X (mm)', ylabel='Object Y (mm)')
    axes[1, 1].set_aspect('equal', adjustable='box')
    axes[1, 1].legend(markerscale=3)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    output = args.output or args.run_dir/'diagnostics.png'
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == '__main__':
    main()
