import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=300)

# ── 构造曲线：正向小峰 → 大负谷 → 缓慢正向反弹衰减至零
# 用三个高斯/指数分量叠加
t = np.linspace(0, 6.0, 800)

# 分量1：早期正向小峰
c1 =  0.30 * np.exp(-((t - 0.45)**2) / (2 * 0.18**2))
# 分量2：主负向深谷
c2 = -1.00 * np.exp(-((t - 1.30)**2) / (2 * 0.45**2))
# 分量3：缓慢正向反弹，指数衰减
c3 =  0.32 * np.exp(-(t - 2.2) / 1.4) * (t > 2.2)

h = c1 + c2 + c3

# 起点和终点钳制为 0
h[0] = 0.0
h[-1] = 0.0

# ── 坐标范围 ──
XMIN, XMAX = -0.5, 6.2
YMIN, YMAX = -1.15, 0.55

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)

# ── 零线 ──
ax.axhline(0, color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=1)

# ── 主曲线（绿色，与参考图一致）──
ax.plot(t, h, color='#1a8a3a', linewidth=2.4, zorder=3)

# ── 隐藏默认 spines ──
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# ── 带箭头坐标轴（无标签，无原点标注）──
ax.annotate('', xy=(XMAX, 0), xytext=(XMIN, 0),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a',
                            lw=1.2, mutation_scale=10))
ax.annotate('', xy=(0, YMAX), xytext=(0, YMIN),
            arrowprops=dict(arrowstyle='->', color='#1a1a1a',
                            lw=1.2, mutation_scale=10))

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout(pad=0.3)

plt.savefig('/mnt/user-data/outputs/post_spike_filter_v2.svg',
            bbox_inches='tight', facecolor='white', format='svg')
plt.savefig('/mnt/user-data/outputs/post_spike_filter_v2.pdf',
            bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/post_spike_filter_v2.png',
            bbox_inches='tight', facecolor='white', dpi=300)
print("Saved!")
