"""Fig 3: Accuracy vs. parameters comparison — 3 backbones + ensemble.

Run on HF Spaces or local with `python paper/scripts/fig3_accuracy.py`.
Only uses known accuracy values — no model inference needed.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def main():
    models = ["MobileNetV2\n(2.4M)", "EfficientNetV2S\n(21.0M)",
              "ConvNeXtTiny\n(28.3M)", "Ensemble (ours)\n(49.3M)"]
    top1 = [88.05, 89.92, 90.78, 92.42]
    top3 = [96.80, 98.67, 98.75, 99.30]
    top5 = [97.89, 99.45, 99.53, 99.69]
    params = [2.4, 21.0, 28.3, 49.3]
    colors = ['#999', '#4ecca3', '#4ecca3', '#e94560']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart
    x = range(len(models))
    w = 0.22
    ax1.bar([i - w for i in x], top1, w, label='Top-1', color='#e94560', edgecolor='white')
    ax1.bar(x, top3, w, label='Top-3', color='#ffd369', edgecolor='white')
    ax1.bar([i + w for i in x], top5, w, label='Top-5', color='#4ecca3', edgecolor='white')

    for i, v in enumerate(top1):
        ax1.text(i - w, v + 0.3, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold', color='#e94560')
    for i, v in enumerate(top3):
        ax1.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7, color='#ffd369')
    for i, v in enumerate(top5):
        ax1.text(i + w, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7, color='#4ecca3')

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_ylim(84, 101)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_title('Classification Accuracy by Model', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Right: params vs accuracy scatter
    ax2.scatter(params[:3], top1[:3], s=[p * 30 for p in params[:3]], c=colors[:3],
                edgecolors='white', linewidth=1.5, zorder=3)
    ax2.scatter([params[3]], [top1[3]], s=params[3] * 30, c='#e94560',
                edgecolors='white', linewidth=2.5, zorder=4, marker='D')
    for i, (p, t) in enumerate(zip(params, top1)):
        ax2.annotate(f'{models[i].strip()}\n({t:.1f}%)', (p, t),
                     textcoords="offset points", xytext=(12, 8 if i < 3 else -18),
                     fontsize=8, fontweight='bold' if i == 3 else 'normal',
                     color=colors[i])
    ax2.set_xlabel('Parameters (millions)', fontsize=11)
    ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax2.set_title('Parameter Efficiency: Accuracy vs. Size', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(87, 93.5)

    plt.tight_layout()
    plt.savefig('paper/figures/fig3_accuracy_vs_params.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0f0f1a')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    plt.savefig('paper/figures/fig3_accuracy_vs_params_dark.png', dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')

    print("Fig 3 saved: paper/figures/fig3_accuracy_vs_params.png")
    print("              paper/figures/fig3_accuracy_vs_params_dark.png")


if __name__ == '__main__':
    main()
