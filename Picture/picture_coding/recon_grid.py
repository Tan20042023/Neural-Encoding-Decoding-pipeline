import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================
# 字体
# ============================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

# ============================================================
# 字号
# ============================================================
FS_ROW_LABEL = 18   # 行标签

# ============================================================
# 配置
# ============================================================
NUM_COLS = 8
NUM_ROWS = 4   # 原始 + 3种重建

row_labels = [
    'Ground Truth',
    '零样本迁移',
    '少样本迁移',
    '编码增强迁移',
]

# 用随机灰度/彩色图模拟（本地替换为真实图像加载逻辑）
rng = np.random.default_rng(42)
def mock_image(color=False):
    if color:
        return rng.integers(60, 220, (64, 64, 3), dtype=np.uint8)
    else:
        return rng.integers(30, 200, (64, 64), dtype=np.uint8)

images = []
for row in range(NUM_ROWS):
    row_imgs = [mock_image(color=(row == 0)) for _ in range(NUM_COLS)]
    images.append(row_imgs)

# ============================================================
# 布局
#
# 思路：整张图固定宽度 14 英寸（接近 A4 横版）
#   左侧 left_margin 留给行标签（figure 坐标）
#   右侧是 4×8 的纯图像网格，wspace=hspace=0（完全紧贴）
#   原始图像行与重建行之间用 hspace 做轻微分隔
# ============================================================
LEFT_MARGIN = 0.11   # figure 宽度的比例，留给行标签
IMG_REGION_LEFT  = LEFT_MARGIN + 0.01
IMG_REGION_RIGHT = 0.995
IMG_REGION_TOP   = 0.985
IMG_REGION_BOT   = 0.015

# 原始行高度略大（彩色图），其余三行等高
# 用 GridSpec height_ratios 控制
HEIGHT_RATIOS = [1.0, 1.0, 1.0, 1.0]
# 原始行和重建行之间加一点间距来做视觉分隔

fig = plt.figure(figsize=(14, 5.0), dpi=150)

gs = GridSpec(
    NUM_ROWS, NUM_COLS,
    figure=fig,
    left=IMG_REGION_LEFT,
    right=IMG_REGION_RIGHT,
    top=IMG_REGION_TOP,
    bottom=IMG_REGION_BOT,
    wspace=0.0,
    hspace=0.03,
    height_ratios=HEIGHT_RATIOS,
)

for row_idx in range(NUM_ROWS):
    for col_idx in range(NUM_COLS):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        img = images[row_idx][col_idx]
        cmap = None if img.ndim == 3 else 'gray'
        ax.imshow(img, cmap=cmap, aspect='auto', interpolation='lanczos')
        ax.axis('off')

        # 图像边框（细线，视觉整洁）
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor('#AAAAAA')

# ============================================================
# 行标签：用 fig.text() 放在 figure 坐标，垂直居中对齐每行
# ============================================================
# 先计算每行在 figure 坐标中的垂直中心
# GridSpec 的行位置可以从 SubplotSpec 获取

def row_vcenter(gs, row_idx, fig):
    """返回某行子图在 figure 坐标中的垂直中心"""
    ss = gs[row_idx, 0]
    bbox = ss.get_position(fig)   # Bbox in figure fraction
    return (bbox.y0 + bbox.y1) / 2

for row_idx, label in enumerate(row_labels):
    yc = row_vcenter(gs, row_idx, fig)

    # 原始图像行字号略大且加一点颜色区分
    if row_idx == 0:
        fs    = FS_ROW_LABEL + 1
        color = '#111111'
        style = 'normal'
    else:
        fs    = FS_ROW_LABEL
        color = '#222222'
        style = 'normal'

    fig.text(
        LEFT_MARGIN - 0.008,   # x 位置：标签列右对齐
        yc,
        label,
        ha='right', va='center',
        fontsize=fs, fontweight='bold',
        fontstyle=style, color=color,
    )

# ============================================================
# 在原始图像行和第一行重建图像之间画分隔线
# ============================================================
# 获取第 0 行底部和第 1 行顶部的 figure y 坐标
bbox_row0 = gs[0, 0].get_position(fig)
bbox_row1 = gs[1, 0].get_position(fig)
sep_y = (bbox_row0.y0 + bbox_row1.y1) / 2   # 两行之间的中线

line = mpatches.FancyArrowPatch(
    (IMG_REGION_LEFT, sep_y), (IMG_REGION_RIGHT, sep_y),
    arrowstyle='-', color='#888888', linewidth=0.8,
    transform=fig.transFigure, clip_on=False
)
fig.add_artist(line)

plt.savefig('/mnt/user-data/outputs/recon_grid.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/mnt/user-data/outputs/recon_grid.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved!")
