# -*- coding: utf-8 -*-
"""色分析 Webアプリ(Pyodide)用の分析モジュール。
ColorAnalysis2.py のブラウザ向け移植版:
 - 画像I/OはPillow経由 (Pyodide の OpenCV は画像コーデックを含まないため)
 - 時間のかかる色置き換え工程 (colorReplace) は含まない
 - 日本語フォントは /fonts/NotoSansJP-Regular.otf を登録して使用
"""
import os
import csv
import warnings

import numpy as np
import cv2
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.font_manager as fm
from PIL import Image as PILImage

import colour
warnings.filterwarnings('ignore', category=colour.utilities.ColourUsageWarning)

FONT_PATH = "/fonts/NotoSansJP-Regular.otf"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = 'Noto Sans JP'

prename = "out"


# ---------- 基本ユーティリティ ----------

def imread_bgr(path):
    pil = PILImage.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def imwrite_bgr(path, img):
    if img.ndim == 2:
        PILImage.fromarray(img).save(path)
    else:
        PILImage.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)).save(path)

def quckresize(img, height):
    return cv2.resize(img, (height, int(height * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_NEAREST)

def returnPointSize(length):
    return 500 if length < 10 else 10

def normPixelColors(testimg):
    pixel_colors = testimg.reshape((np.shape(testimg)[0] * np.shape(testimg)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    return norm(pixel_colors).tolist()


# ---------- 描画部品 (単体画像とサマリーシートで共用) ----------

def draw_rgb3d(axis, testimg, pixel_colors):
    r, g, b = cv2.split(testimg)
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".", s=returnPointSize(r.flatten().shape[0]))
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")

def draw_hsv3d(axis, testimg, pixel_colors):
    h, s, v = cv2.split(cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV))
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".", s=returnPointSize(h.flatten().shape[0]))
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

def draw_scatter2d(axis, x, y, pixel_colors, psize, xlabel, ylabel):
    axis.scatter(x, y, facecolors=pixel_colors, marker=".", s=psize)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

def draw_labcircle(axis, testimg, pixel_colors):
    lab = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] *= 100 / 255
    lab[:, :, 1] -= 128
    lab[:, :, 2] -= 128
    l, a, b = cv2.split(lab)
    axis.hlines(0, -128, 128, colors=["#00000055"])
    axis.vlines(0, -128, 128, colors=["#00000055"])
    t = np.linspace(0., 8.0, 100)
    axis.plot(np.sin(t) * 64, np.cos(t) * 64, c='#00000055')
    axis.plot(np.sin(t) * 127, np.cos(t) * 127, c='#00000055')
    axis.plot(np.sin(t) * 32, np.cos(t) * 32, c='#00000022')
    axis.plot(np.sin(t) * 96, np.cos(t) * 96, c='#00000022')
    axis.scatter(b.flatten(), a.flatten(), facecolors=pixel_colors, marker=".", s=returnPointSize(b.flatten().shape[0]))
    axis.set_xlabel("b* (Yellow - blue)")
    axis.set_ylabel("a* (green - red)")


# ---------- 単体マップ画像の書き出し ----------

def rgbmap(testimg):
    pixel_colors = normPixelColors(testimg)
    fig = plt.figure(facecolor='#b7b7b700')
    axis = fig.add_subplot(1, 1, 1, projection="3d", facecolor='#b7b7b700')
    draw_rgb3d(axis, testimg, pixel_colors)
    fig.savefig(prename + '_RGBspace_map.png', facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)

def hsvmap(testimg):
    pixel_colors = normPixelColors(testimg)
    h, s, v = cv2.split(cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV))
    psize = returnPointSize(h.flatten().shape[0])

    fig = plt.figure(facecolor='#b7b7b700')
    axis = fig.add_subplot(1, 1, 1, projection="3d", facecolor='#b7b7b700')
    draw_hsv3d(axis, testimg, pixel_colors)
    fig.savefig(prename + '_HSVspace_map.png', facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)

    for x, y, xl, yl, xlim, ylim, fname in [
            (h, v, "Hue", "Value", (0, 180), (0, 255), '_HSV_hue-val_map.png'),
            (s, v, "Saturation", "Value", (0, 255), (0, 255), '_HSV_sat-val_map.png'),
            (h, s, "Hue", "Saturation", (0, 180), (0, 255), '_HSV_hue-sat_map.png')]:
        fig = plt.figure(facecolor='#ffffff00')
        axis = fig.add_subplot(111, facecolor='#ffffff00', xlim=xlim, ylim=ylim)
        draw_scatter2d(axis, x.flatten(), y.flatten(), pixel_colors, psize, xl, yl)
        fig.savefig(prename + fname, facecolor=fig.get_facecolor(), dpi=200)
        plt.close(fig)

def labmap(testimg):
    pixel_colors = normPixelColors(testimg)
    fig = plt.figure(figsize=(6, 6), facecolor='#b7b7b700')
    axis = fig.add_axes((0.12, 0.09, 0.8, 0.8), facecolor='#b7b7b700', xlim=(-128, 128), ylim=(-128, 128))
    draw_labcircle(axis, testimg, pixel_colors)
    fig.savefig(prename + '_Lab-circle_map.png', facecolor=fig.get_facecolor(), dpi=200)
    plt.close(fig)


# ---------- K平均法 ----------

def kmeanmap(testimg, colnum):
    data = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    K = colnum
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
    ret, label, center = cv2.kmeans(data, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(label, return_counts=True)

    # 各画素を代表色に置き換えた画像
    dst = center[label.ravel()].reshape(testimg.shape).astype(np.uint8)
    imwrite_bgr(prename + "kmean(lab)-replace_" + str(K) + "colors.png", cv2.cvtColor(dst, cv2.COLOR_LAB2BGR))

    RGBtmp = cv2.cvtColor(center.astype(np.uint8).reshape(1, K, 3), cv2.COLOR_LAB2RGB)
    HSVtmp = cv2.cvtColor(RGBtmp, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    RGBtmp16str = ["#{:02x}{:02x}{:02x}".format(*c) for c in RGBtmp.reshape(-1, 3)]

    with open(prename + "kmean(lab)-replace_" + str(K) + "colors.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["RGB", "Hue", "Saturation", "Value", "Area"])
        for i in range(len(RGBtmp16str)):
            writer.writerow([RGBtmp16str[i], HSVtmp[i, 0], HSVtmp[i, 1], HSVtmp[i, 2], counts[i]])

    # ColorBar のレンダリング (Hue/Saturation/Value/Area 順)
    barWidth, barHeight = 1000, 100
    strset = ["Hue", "Saturation", "Value", "Area"]
    scalenum = barWidth / sum(counts)
    for j in range(4):
        newOrder = np.argsort(HSVtmp[:, j]) if j != 3 else np.argsort(counts)
        newimg = None
        for i in newOrder:
            vol = max(1, int(counts[i] * scalenum))
            block = np.full((barHeight, vol, 3), center.astype(np.uint8)[i])
            newimg = block if newimg is None else np.hstack((newimg, block))
        imwrite_bgr(prename + "kmean(lab)-color_Bar(" + strset[j] + ")-" + str(K) + "colors.png",
                    cv2.cvtColor(newimg, cv2.COLOR_LAB2BGR))

    return center.astype(np.uint8), counts


# ---------- マンセル変換 ----------
# 変換理論 (ASTM D1535 方式):
#  1. sRGB を逆ガンマ補正して線形RGBへ戻し、3x3行列で CIE XYZ (D65光源基準) に変換
#  2. マンセル表色系の測定基準である C光源 へ色順応変換 (CAT02)
#  3. XYZ → xyY (色度座標 x,y + 輝度 Y) に変換
#  4. Munsell Renotation Data (1943年の視覚実験による対応表) を ASTM D1535 に基づき補間し、
#     xyY からマンセル値 (色相 H, 明度 V, 彩度 C) を逆算

def rgb2munsell(rgb):
    rgb01 = np.asarray(rgb, dtype=np.float64) / 255.0
    XYZ = colour.sRGB_to_XYZ(rgb01)
    obs = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
    XYZ_C = colour.adaptation.chromatic_adaptation(
        XYZ, colour.xy_to_XYZ(obs['D65']), colour.xy_to_XYZ(obs['C']))
    xyY = colour.XYZ_to_xyY(XYZ_C)
    try:
        notation = colour.notation.munsell.xyY_to_munsell_colour(xyY)
    except Exception:
        V = float(colour.notation.munsell.munsell_value_ASTMD1535(xyY[2] * 100))
        return "N {:.1f}".format(V), None, V, 0.0
    spec = colour.notation.munsell.munsell_colour_to_munsell_specification(notation)
    if np.isnan(spec[0]):
        return notation, None, float(spec[1]), 0.0
    hue_angle = float(colour.notation.munsell.hue_to_ASTM_hue([spec[0], spec[3]])) * 3.6
    return notation, hue_angle, float(spec[1]), float(spec[2])

def munsellEntriesFromCenters(labCenters, counts):
    K = labCenters.shape[0]
    rgbs = cv2.cvtColor(labCenters.reshape(1, K, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    entries = []
    for rgb, cnt in zip(rgbs, counts):
        notation, hue_angle, mv, mc = rgb2munsell(rgb)
        entries.append({"rgb01": tuple(rgb / 255.0), "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                        "notation": notation, "angle": hue_angle,
                        "value": mv, "chroma": mc, "count": int(cnt)})
    entries.sort(key=lambda e: (e["angle"] is not None, e["angle"] if e["angle"] is not None else 0))
    return entries, sum(e["count"] for e in entries)

def draw_munsell_grid(axp, rmax):
    hue_angles, hue_labels = [], []
    for fam in ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']:
        for step in ['2.5', '5.0', '7.5', '10.0']:
            spec = colour.notation.munsell.munsell_colour_to_munsell_specification(step + fam + " 5.0/10.0")
            hue_angles.append(float(colour.notation.munsell.hue_to_ASTM_hue([spec[0], spec[3]])) * 3.6)
            hue_labels.append(step.rstrip('0').rstrip('.') + fam)
    axp.set_thetagrids(hue_angles, hue_labels, fontsize=8)
    for lbl in axp.get_xticklabels():
        if lbl.get_text().startswith('5'):
            lbl.set_fontsize(11)
            lbl.set_fontweight('bold')
    axp.set_rlim(0, rmax)
    axp.set_rgrids(np.arange(4, rmax + 0.1, 4), fontsize=8, color='#00000088')
    axp.grid(color='#00000022', linewidth=0.4)

def draw_munsell_circle(axp, entries, total, rmax):
    draw_munsell_grid(axp, rmax)
    angles = [0 if e["angle"] is None else e["angle"] for e in entries]
    axp.scatter(np.radians(angles), [e["chroma"] for e in entries],
                c=[e["rgb01"] for e in entries],
                s=[120 + 1200 * e["count"] / total for e in entries],
                edgecolors='#00000055', zorder=3)

_munsell_cache = {}

def munsellPixelData(testimg, max_unique=600):
    """全画素をマンセル(色相角, 彩度)へ変換する。
    厳密な逆算は1色あたり数十msかかるため、Labを量子化して固有色ごとに変換し
    (結果はキャッシュ)、固有色数が max_unique 以下になる最小の量子化幅を自動選択する。"""
    lab = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    for shift in (2, 3, 4, 5):
        q = (lab >> shift) << shift
        uniq, inv = np.unique(q, axis=0, return_inverse=True)
        if uniq.shape[0] <= max_unique:
            break
    rgbs = cv2.cvtColor(uniq.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    angles = np.zeros(uniq.shape[0])
    chromas = np.zeros(uniq.shape[0])
    for i, rgb in enumerate(rgbs):
        key = tuple(rgb)
        if key not in _munsell_cache:
            notation, ang, mv, mc = rgb2munsell(rgb)
            _munsell_cache[key] = (0.0 if ang is None else ang, mc)
        angles[i], chromas[i] = _munsell_cache[key]
    return angles[inv.ravel()], chromas[inv.ravel()]

def draw_value_chroma(axv, entries, total, rmax):
    axv.set_xlim(0, rmax)
    axv.set_ylim(0, 10)
    axv.set_xticks(np.arange(0, rmax + 0.1, 2))
    axv.set_xticks(np.arange(0, rmax + 0.1, 1), minor=True)
    axv.set_yticks(np.arange(0, 10.1, 1))
    axv.grid(which='both', color='#00000022')
    # グラフ端の色でもマーカーが半円にクリップされないよう clip_on=False
    axv.scatter([e["chroma"] for e in entries], [e["value"] for e in entries],
                c=[e["rgb01"] for e in entries],
                s=[120 + 1200 * e["count"] / total for e in entries],
                edgecolors='#00000055', zorder=3, clip_on=False)
    axv.set_xlabel("Chroma (彩度)")
    axv.set_ylabel("Value (明度)")


# ---------- サマリーシート ----------

def savefigRaster(fig, basepath, dpi=200):
    fig.savefig(basepath + ".png", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    PILImage.open(basepath + ".png").convert("RGB").save(basepath + ".pdf", resolution=dpi)

def makeSummarySheet(raw_img, testimg, title, author, student, labCenters, counts):
    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#b7b7b7')

    def underline(x0, x1, y):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color='black', linewidth=0.8, transform=fig.transFigure))

    pixel_colors = normPixelColors(testimg)
    h, s, v = cv2.split(cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV))
    psize = returnPointSize(h.flatten().shape[0])

    fig.text(0.03, 0.955, "氏名", fontsize=11)
    fig.text(0.065, 0.955, student, fontsize=11)
    underline(0.06, 0.24, 0.952)

    ax = fig.add_axes([0.025, 0.55, 0.155, 0.38], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    ax = fig.add_axes([0.19, 0.55, 0.155, 0.38], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), cmap='gray')

    ax3 = fig.add_axes([0.005, 0.235, 0.19, 0.30], projection='3d', facecolor='#b7b7b700')
    draw_rgb3d(ax3, testimg, pixel_colors)
    fig.text(0.10, 0.185, "RGB色空間マッピング", fontsize=9, ha='center')
    ax3 = fig.add_axes([0.18, 0.235, 0.19, 0.30], projection='3d', facecolor='#b7b7b700')
    draw_hsv3d(ax3, testimg, pixel_colors)
    fig.text(0.275, 0.185, "HSV色空間マッピング", fontsize=9, ha='center')

    fig.text(0.42, 0.945, "HSV空間マッピング", fontsize=13)
    for x, y, xl, yl, xlim, ylim, rect in [
            (h, v, "Hue", "Value", (0, 180), (0, 255), [0.415, 0.745, 0.145, 0.17]),
            (h, s, "Hue", "Saturation", (0, 180), (0, 255), [0.415, 0.49, 0.145, 0.17]),
            (s, v, "Saturation", "Value", (0, 255), (0, 255), [0.415, 0.235, 0.145, 0.17])]:
        axs = fig.add_axes(rect, facecolor='#ffffff00', xlim=xlim, ylim=ylim)
        draw_scatter2d(axs, x.flatten(), y.flatten(), pixel_colors, psize, xl, yl)

    # マンセル色相環: 1枚目は全画素マッピング (2枚目は代表色マッピング)
    pangles, pchromas = munsellPixelData(testimg)
    rmax = max(14.0, float(np.max(pchromas)) + 1)
    fig.text(0.66, 0.945, "マンセル色相環マッピング (Hue / Chroma)", fontsize=13)
    axp = fig.add_axes([0.62, 0.28, 0.34, 0.62], projection='polar', facecolor='#ffffff00')
    draw_munsell_grid(axp, rmax)
    axp.scatter(np.radians(pangles), pchromas, facecolors=pixel_colors, marker=".", s=psize*4, zorder=3)

    K = labCenters.shape[0]
    fig.text(0.975, 0.225, "K平均法による代表色の抽出と面積比", fontsize=10, ha='right')
    for label, y in zip(["Hue", "Saturation", "Value"], [0.145, 0.085, 0.025]):
        fig.text(0.345, y + 0.052, label, fontsize=9)
        bar = plt.imread(prename + "kmean(lab)-color_Bar(" + label + ")-" + str(K) + "colors.png")
        ax = fig.add_axes([0.345, y, 0.63, 0.05])
        ax.set_axis_off()
        ax.imshow(bar, aspect='auto')

    fig.text(0.03, 0.095, "タイトル", fontsize=11)
    fig.text(0.095, 0.095, title, fontsize=11)
    underline(0.09, 0.30, 0.09)
    fig.text(0.03, 0.04, "作者", fontsize=11)
    fig.text(0.095, 0.04, author, fontsize=11)
    underline(0.09, 0.30, 0.035)

    savefigRaster(fig, prename + "_summary")

def makeMunsellSheet(raw_img, title, author, student, K=10):
    testimg = cv2.cvtColor(quckresize(raw_img, 100), cv2.COLOR_BGR2RGB)
    data = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
    ret, label, center = cv2.kmeans(data, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(label, return_counts=True)
    entries, total = munsellEntriesFromCenters(center.astype(np.uint8), counts)

    with open(prename + "_munsell_" + str(K) + "colors.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["RGB(hex)", "Munsell", "HueAngle(deg)", "Value", "Chroma", "Area"])
        for e in entries:
            writer.writerow([e["hex"], e["notation"],
                             "" if e["angle"] is None else round(e["angle"], 1),
                             e["value"], e["chroma"], e["count"]])

    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#b7b7b7')

    def underline(x0, x1, y):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color='black', linewidth=0.8, transform=fig.transFigure))

    fig.text(0.03, 0.955, "氏名", fontsize=11)
    fig.text(0.065, 0.955, student, fontsize=11)
    underline(0.06, 0.24, 0.952)
    ax = fig.add_axes([0.025, 0.30, 0.30, 0.60], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    fig.text(0.03, 0.095, "タイトル", fontsize=11)
    fig.text(0.095, 0.095, title, fontsize=11)
    underline(0.09, 0.30, 0.09)
    fig.text(0.03, 0.04, "作者", fontsize=11)
    fig.text(0.095, 0.04, author, fontsize=11)
    underline(0.09, 0.30, 0.035)

    rmax = max(14.0, max(e["chroma"] for e in entries) + 2)

    fig.text(0.42, 0.945, "マンセル色相環マッピング (Hue / Chroma)", fontsize=13)
    axp = fig.add_axes([0.36, 0.36, 0.30, 0.54], projection='polar', facecolor='#ffffff00')
    draw_munsell_circle(axp, entries, total, rmax)

    fig.text(0.75, 0.945, "明度・彩度図 (Value / Chroma)", fontsize=13)
    axv = fig.add_axes([0.735, 0.40, 0.23, 0.48], facecolor='#ffffff00')
    draw_value_chroma(axv, entries, total, rmax)

    fig.text(0.36, 0.30, "K平均法による代表" + str(K) + "色とマンセル値 (面積比)", fontsize=11)
    axb = fig.add_axes([0.36, 0.16, 0.61, 0.10])
    axb.set_axis_off()
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    x = 0.0
    narrow_i = 0
    # カラーバーは面積比の大きい順(左→右)に並べる
    for e in sorted(entries, key=lambda e: -e["count"]):
        w = e["count"] / total
        axb.add_patch(plt.Rectangle((x, 0), w, 1, facecolor=e["rgb01"], edgecolor='none'))
        if w >= 0.03:
            txtcol = 'white' if e["value"] < 5.5 else 'black'
            axb.text(x + w / 2, 0.5, "{:.0f}%".format(w * 100), ha='center', va='center', fontsize=8, color=txtcol)
        # 狭いセグメントが連続するとラベルが重なるため、上下2段に振り分ける
        if w < 0.05:
            ylabel = -0.10 if narrow_i % 2 == 0 else -0.75
            narrow_i += 1
        else:
            ylabel = -0.10
            narrow_i = 0
        axb.text(x + w / 2, ylabel, e["notation"], rotation=40, ha='right', va='top',
                 rotation_mode='anchor', fontsize=9, clip_on=False)
        x += w

    fig.text(0.36, 0.03,
             "RGB→マンセル変換: sRGB →(逆ガンマ補正)→ CIE XYZ (D65) →(色順応変換 CAT02: D65→C光源)→ xyY\n"
             "→ Munsell Renotation Data (ASTM D1535) の補間により 色相H・明度V・彩度C を算出",
             fontsize=9, va='bottom')

    savefigRaster(fig, prename + "_summary2")


# ---------- エントリポイント (JSから呼ばれる) ----------

def run_analysis(img_path, out_root, title="", author="", student=""):
    """1枚の画像を分析し、out_root/<画像名>color-analysis/ に全成果物を書き出す。"""
    global prename
    raw_img = imread_bgr(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    if not title:
        title = os.path.basename(img_path)

    out_dir = os.path.join(out_root, base + "color-analysis")
    os.makedirs(out_dir, exist_ok=True)
    prename = os.path.join(out_dir, base)

    imwrite_bgr(prename + "_gray.png", cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY))

    testimg = cv2.cvtColor(quckresize(raw_img, 100), cv2.COLOR_BGR2RGB)

    rgbmap(testimg)
    hsvmap(testimg)
    labmap(testimg)
    centers, counts = kmeanmap(testimg, 20)
    makeSummarySheet(raw_img, testimg, title, author, student, centers, counts)
    makeMunsellSheet(raw_img, title, author, student, 10)
    plt.close('all')
    return out_dir
