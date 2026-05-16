import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec

# ============================================================
# 字体设置
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.major.size': 4,    'ytick.major.size': 4,
})

# ============================================================
# 模拟数据（本地替换为真实 load_aggregated_metrics 逻辑）
# ============================================================
noise_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

def make_data(noise_type):
    rng = np.random.default_rng(42)
    n = len(noise_levels)
    if noise_type == 'Dropout':
        mse_true  = 0.005 + 0.038 * noise_levels**1.6 + rng.normal(0, 0.001, n)
        mse_pred  = 0.004 + 0.030 * noise_levels**1.6 + rng.normal(0, 0.001, n)
        psnr_true = 23.5  - 10   * noise_levels**1.2  + rng.normal(0, 0.2, n)
        psnr_pred = 24.0  - 8.5  * noise_levels**1.2  + rng.normal(0, 0.2, n)
        ssim_true = 0.72  - 0.52 * noise_levels**1.1  + rng.normal(0, 0.01, n)
        ssim_pred = 0.74  - 0.46 * noise_levels**1.1  + rng.normal(0, 0.01, n)
    elif noise_type == 'Gaussian':
        mse_true  = 0.005 + 0.042 * noise_levels**1.4 + rng.normal(0, 0.001, n)
        mse_pred  = 0.004 + 0.032 * noise_levels**1.4 + rng.normal(0, 0.001, n)
        psnr_true = 23.5  - 11   * noise_levels**1.1  + rng.normal(0, 0.2, n)
        psnr_pred = 24.0  - 9.0  * noise_levels**1.1  + rng.normal(0, 0.2, n)
        ssim_true = 0.72  - 0.54 * noise_levels**1.0  + rng.normal(0, 0.01, n)
        ssim_pred = 0.74  - 0.44 * noise_levels**1.0  + rng.normal(0, 0.01, n)
    else:  # Poisson
        mse_true  = 0.005 + 0.010 * noise_levels**1.8 + rng.normal(0, 0.0005, n)
        mse_pred  = 0.004 + 0.008 * noise_levels**1.8 + rng.normal(0, 0.0005, n)
        psnr_true = 23.5  - 3.5  * noise_levels**1.0  + rng.normal(0, 0.1, n)
        psnr_pred = 24.0  - 2.8  * noise_levels**1.0  + rng.normal(0, 0.1, n)
        ssim_true = 0.74  - 0.15 * noise_levels**1.0  + rng.normal(0, 0.005, n)
        ssim_pred = 0.75  - 0.12 * noise_levels**1.0  + rng.normal(0, 0.005, n)
    return {
        'noise_levels': noise_levels,
        'noisy_spike_decode':      {'mse': mse_true,  'psnr': psnr_true,  'ssim': ssim_true},
        'noisy_pred_spike_decode': {'mse': mse_pred,  'psnr': psnr_pred,  'ssim': ssim_pred},
    }

all_data = {k: make_data(k) for k in ['Dropout', 'Gaussian', 'Poisson']}

# ============================================================
# 字号
# ============================================================
FS_COL_TITLE  = 20
FS_ROW_LABEL  = 18
FS_AXIS_LABEL = 16
FS_TICK       = 14
FS_PANEL      = 16
FS_LEGEND     = 18

# ============================================================
# 配置
# ============================================================
row_labels = {'Dropout': '失活噪声', 'Gaussian': '高斯噪声', 'Poisson': '泊松噪声'}

metric_configs = {
    'mse':  {'title': 'MSE',  'ylabel': 'MSE (×10$^{-3}$)',
              'formatter': FuncFormatter(lambda v, p: f'{v*1e3:.1f}')},
    'psnr': {'title': 'PSNR', 'ylabel': 'PSNR (dB)',
              'formatter': FuncFormatter(lambda v, p: f'{v:.1f}')},
    'ssim': {'title': 'SSIM', 'ylabel': 'SSIM',
              'formatter': FuncFormatter(lambda v, p: f'{v:.2f}')},
}

series_labels = {
    'noisy_spike_decode':      '噪声真实脉冲解码',
    'noisy_pred_spike_decode': '噪声预测脉冲解码',
}

line_styles = {
    'noisy_spike_decode': {
        'color': '#0072B2', 'marker': 'o', 'linestyle': '-',
        'linewidth': 2.2, 'markersize': 6.5,
        'markerfacecolor': '#0072B2', 'markeredgecolor': 'black', 'markeredgewidth': 0.7,
    },
    'noisy_pred_spike_decode': {
        'color': '#D55E00', 'marker': 's', 'linestyle': '--',
        'linewidth': 2.2, 'markersize': 6.5,
        'markerfacecolor': '#D55E00', 'markeredgecolor': 'black', 'markeredgewidth': 0.7,
    },
}

metric_ylim = {'mse': (0.0035, 0.0395), 'psnr': (14.0, 24.8), 'ssim': (0.25, 0.79)}
noise_order  = ['Dropout', 'Gaussian', 'Poisson']
metric_order = ['mse', 'psnr', 'ssim']
panel_labels = list('abcdefghi')
x_ticks      = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# ============================================================
# 布局
#   left=0.17  → 给 y轴刻度 + y轴标签 + 行标签三层留足空间
#   bottom=0.12 → 给 x刻度 + "噪声强度" + 子图编号 + 图例四层留足空间
#   wspace=0.18 → 列间紧凑（只有刻度数字，无重复y轴标签）
#   hspace=0.20 → 行间紧凑
# ============================================================
fig = plt.figure(figsize=(16, 12), dpi=150)
gs = GridSpec(3, 3, figure=fig,
              left=0.17,    # ← 加宽，给行标签 + y轴标签各自留空间
              right=0.97,
              top=0.94,
              bottom=0.13,  # ← 加高，给子图编号 + x轴标签 + 图例留空间
              wspace=0.18,  # ← 列间收紧（无重复y轴标签后可以更紧）
              hspace=0.20)

first_handles = None
panel_index   = 0

for row_idx, noise_name in enumerate(noise_order):
    data = all_data[noise_name]

    for col_idx, metric in enumerate(metric_order):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        y_true = data['noisy_spike_decode'][metric]
        y_pred = data['noisy_pred_spike_decode'][metric]

        for key, y in [('noisy_spike_decode', y_true), ('noisy_pred_spike_decode', y_pred)]:
            ls = line_styles[key]
            ax.plot(noise_levels, y,
                    marker=ls['marker'], linestyle=ls['linestyle'],
                    color=ls['color'], linewidth=ls['linewidth'],
                    markersize=ls['markersize'],
                    markerfacecolor=ls['markerfacecolor'],
                    markeredgecolor=ls['markeredgecolor'],
                    markeredgewidth=ls['markeredgewidth'],
                    label=series_labels[key], zorder=3)

        ax.set_ylim(*metric_ylim[metric])
        ax.yaxis.set_major_formatter(metric_configs[metric]['formatter'])
        ax.grid(True, linestyle=':', alpha=0.4, color='#CCCCCC')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=FS_TICK, direction='out')

        # ── 列标题：仅第一行 ──
        if row_idx == 0:
            ax.set_title(metric_configs[metric]['title'],
                         fontsize=FS_COL_TITLE, fontweight='bold', pad=10)

        # ── y 轴标签：仅第一列；其余列只保留刻度数字 ──
        if col_idx == 0:
            ax.set_ylabel(metric_configs[metric]['ylabel'],
                          fontsize=FS_AXIS_LABEL, fontweight='bold', labelpad=6)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        # ── x 轴刻度：仅最后一行显示数字和"噪声强度" ──
        ax.set_xticks(x_ticks)
        if row_idx == 2:
            ax.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=FS_TICK)
            # "噪声强度" labelpad 加大，与子图编号之间留出安全距离
            ax.set_xlabel('噪声强度', fontsize=FS_AXIS_LABEL, fontweight='bold', labelpad=28)
        else:
            ax.set_xticklabels([])

        # ── 行标签：竖排，放在 y轴标签 更左侧，两者不重叠 ──
        # -0.38 在 axes 坐标下足够推到 y轴标签左边
        if col_idx == 0:
            ax.text(-0.38, 0.5, row_labels[noise_name],
                    transform=ax.transAxes,
                    rotation=90, va='center', ha='center',
                    fontsize=FS_ROW_LABEL, fontweight='bold', color='#222222')

        # ── 子图编号：x轴正下方居中，固定 axes 坐标 -0.08
        #    最后一行偏移更大（-0.08）以跳过 x刻度数字；
        #    "噪声强度" 用 labelpad=28 推到编号下方，两者不重叠
        panel_offset = -0.08   # 对所有行统一，跳过 x 刻度层
        ax.text(0.5, panel_offset,
                f'({panel_labels[panel_index]})',
                transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight='bold',
                va='top', ha='center', color='black')
        panel_index += 1

        if first_handles is None:
            first_handles, _ = ax.get_legend_handles_labels()

# ============================================================
# 图例：图正下方
# ============================================================
fig.legend(
    first_handles,
    [series_labels['noisy_spike_decode'], series_labels['noisy_pred_spike_decode']],
    loc='lower center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.0),
    fontsize=FS_LEGEND,
    handlelength=3.2,
    handletextpad=0.8,
    columnspacing=2.5,
)

plt.savefig('/mnt/user-data/outputs/noise_summary_v4.pdf',
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('/mnt/user-data/outputs/noise_summary_v4.png',
            dpi=150, bbox_inches='tight', format='png')
print("Saved!")
