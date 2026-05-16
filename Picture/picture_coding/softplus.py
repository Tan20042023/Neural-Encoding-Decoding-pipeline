import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=300)

# ── 数据 ──
x = np.linspace(-3.5, 3.5, 400)
softplus = np.log1p(np.exp(x))          # softplus: ln(1 + e^x)
relu     = np.maximum(0, x)             # ReLU

# ── 曲线 ──
ax.plot(x, softplus, color='#1a1a1a', linewidth=2.2, zorder=3, label='Softplus')
ax.plot(x, relu,     color='#1a1a1a', linewidth=1.4,
        linestyle='--', dashes=(5, 3), zorder=2, alpha=0.55, label='ReLU')

# ── 坐标轴范围 ──
XMIN, XMAX = -3.2, 3.5
YMIN, YMAX = -0.5, 3.8
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)

# ── 隐藏默认 spines ──
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# ── 手绘风格的带箭头坐标轴 ──
arrow_kw = dict(
    arrowstyle='->', color='#1a1a1a', linewidth=1.2,
    mutation_scale=10, clip_on=False,
    transform=ax.transData,
)
# x 轴
ax.annotate('', xy=(XMAX, 0), xytext=(XMIN, 0),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a',
                            lw=1.2, mutation_scale=10))
# y 轴
ax.annotate('', xy=(0, YMAX), xytext=(0, YMIN),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a',
                            lw=1.2, mutation_scale=10))

# 原点标记
ax.text(-0.18, -0.32, '$O$', fontsize=11, ha='right', va='top', color='#1a1a1a')

# 轴标签
ax.text(XMAX + 0.05, -0.05, '$x$', fontsize=13, ha='left', va='top',
        fontstyle='italic', color='#1a1a1a')
ax.text(0.12, YMAX, '$f(x)$', fontsize=12, ha='left', va='bottom', color='#1a1a1a')

# ── 图例：放在右下方，简洁无框 ──
softplus_line = plt.Line2D([0], [0], color='#1a1a1a', linewidth=2.2)
relu_line     = plt.Line2D([0], [0], color='#1a1a1a', linewidth=1.4,
                            linestyle='--', dashes=(5, 3), alpha=0.55)
leg = ax.legend(
    [softplus_line, relu_line],
    ['Softplus', 'ReLU'],
    loc='upper left',
    frameon=False,
    fontsize=10.5,
    handlelength=1.8,
    handletextpad=0.5,
    labelspacing=0.3,
    bbox_to_anchor=(0.04, 0.98),
)

# ── 背景 ──
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout(pad=0.3)

plt.savefig('/mnt/user-data/outputs/softplus.pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/softplus.svg',
            bbox_inches='tight', facecolor='white', format='svg')
plt.savefig('/mnt/user-data/outputs/softplus.png',
            bbox_inches='tight', facecolor='white', dpi=300)
print("Saved!")
