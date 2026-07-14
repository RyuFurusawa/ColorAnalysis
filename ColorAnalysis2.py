import os, glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import csv
import sys
import colour  # colour-science: マンセル変換 (ASTM D1535) に使用
import warnings
#Renotation Data範囲外の暗色などで出る情報通知を抑制(変換自体はフォールバックで処理される)
warnings.filterwarnings('ignore', category=colour.utilities.ColourUsageWarning)
from PIL import Image as PILImage
# from google.colab.patches import cv2_imshow
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(len(flags), flags[40])
prename = "strings"
#labに変換
def labdecord(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #それぞれ8bitに変換されている。
    lab_img2 = lab_img.astype(np.float64)
    lab_img2[:, :, 0] *= 100 / 255  # L の範囲を [0, 100] に戻す。
    lab_img2[:, :, 1] -= 128  # L の範囲を [-127, 127] に戻す。
    lab_img2[:, :, 2] -= 128  # L の範囲を [-127, 127] に戻す。
    return lab_img2

#lab8bitをfloatに変換に変換

def imglab8bit2float(img):
    lab_img = img.astype(np.float64)
    lab_img[:, :, 0] *= 100 / 255  # L の範囲を [0, 100] に戻す。
    lab_img[:, :, 1] -= 128  # L の範囲を [-127, 127] に戻す。
    lab_img[:, :, 2] -= 128  # L の範囲を [-127, 127] に戻す。
    return lab_img

def colorlab8bit2float(color):
    color = color.astype(np.float64)
    color[:,0] *= 100 / 255  # L の範囲を [0, 100] に戻す。
    color[:, 1] -= 128  # L の範囲を [-127, 127] に戻す。
    color[:, 2] -= 128  # L の範囲を [-127, 127] に戻す。
    return color

def colorlabfloat28bit(color):
    # color = color.astype(np.float64)
    color[:,0] *= 255 / 100  
    color[:, 1] += 128  
    color[:, 2] += 128 
    return color.astype(np.uint8)

def trans6lab(input_color):
    colors = np.zeros((input_color.shape[0], 6), np.float64)
    
    colors[:,0] = input_color[:,0] - 50.0
    colors[:,1] = input_color[:,0] - 50.0
    colors[:,2] = input_color[:,1]
    colors[:,3] = input_color[:,1]
    colors[:,4] = input_color[:,2]
    colors[:,5] = input_color[:,2]
    
    colors[:,0]=np.where(colors[:,0] < 0,0,colors[:,0])
    colors[:,1]=np.where(colors[:,1] > 0,0,colors[:,1]*-1)
    colors[:,2]=np.where(colors[:,2] < 0,0,colors[:,2])
    colors[:,3]=np.where(colors[:,3] > 0,0,colors[:,3]*-1)
    colors[:,4]=np.where(colors[:,4] < 0,0,colors[:,4])
    colors[:,5]=np.where(colors[:,5] > 0,0,colors[:,5]*-1)
    return colors

def decord3lab(input_color):
    colors = np.zeros((input_color.shape[0], 3), np.float64)
    colors[:,0] = (input_color[:,0]-input_color[:,1])+50.0
    colors[:,1] = input_color[:,2]-input_color[:,3]
    colors[:,2] = input_color[:,4]-input_color[:,5]
    return colors

def returnPointSize(len):
  if(len < 10):
    psize=500
  else:
    psize=10
  return psize

#散布図の各点に付ける表示色([0,1]正規化RGB)のリストを作る
def normPixelColors(testimg):
    pixel_colors = testimg.reshape((np.shape(testimg)[0]*np.shape(testimg)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    return norm(pixel_colors).tolist()

#以下の draw_ 系関数は指定された axes に直接描画する(単体画像とサマリーシートで共用)
def draw_rgb3d(axis, testimg, pixel_colors):
    r, g, b = cv2.split(testimg)
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".",s=returnPointSize(r.flatten().shape[0]))
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")

def draw_hsv3d(axis, testimg, pixel_colors):
    hsv_testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_testimg)
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".",s=returnPointSize(h.flatten().shape[0]))
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")

def draw_scatter2d(axis, x, y, pixel_colors, psize, xlabel, ylabel):
    axis.scatter(x, y, facecolors=pixel_colors, marker=".", s=psize)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

def draw_labcircle(axis, testimg, pixel_colors):
    lab_testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB)
    lab_testimg = lab_testimg.astype(np.float64)
    lab_testimg[:, :, 0] *= 100 / 255  # L の範囲を [0, 100] に戻す。
    lab_testimg[:, :, 1] -= 128  # L の範囲を [-127, 127] に戻す。
    lab_testimg[:, :, 2] -= 128  # L の範囲を [-127, 127] に戻す。
    l, a, b = cv2.split(lab_testimg)
    #軸線の描画
    axis.hlines(0, -128, 128,colors=["#00000055"])
    axis.vlines(0, -128, 128,colors=["#00000055"])
    t = np.linspace(0., 8.0, 100)
    #同心円の描画
    axis.plot(np.sin(t) * 64 ,np.cos(t) * 64,c='#00000055')
    axis.plot(np.sin(t) * 127,np.cos(t) * 127,c='#00000055')
    axis.plot(np.sin(t) * 32,np.cos(t) * 32,c='#00000022')
    axis.plot(np.sin(t) * 96,np.cos(t) * 96,c='#00000022')
    axis.scatter(b.flatten(),a.flatten(), facecolors=pixel_colors, marker=".",s=returnPointSize(b.flatten().shape[0]))
    axis.set_xlabel("b* (Yellow - blue)")
    axis.set_ylabel("a* (green - red)")

def rgbmap(testimg):
    pixel_colors = normPixelColors(testimg)
    fig = plt.figure(facecolor='#b7b7b700')
    axis = fig.add_subplot(1, 1, 1, projection="3d", facecolor='#b7b7b700')
    draw_rgb3d(axis, testimg, pixel_colors)
    fig.savefig(prename+'_RGBspace_map.png', facecolor=fig.get_facecolor(), dpi=200)


def hsvmap(testimg):
    pixel_colors = normPixelColors(testimg)
    hsv_testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_testimg)
    psize=returnPointSize(h.flatten().shape[0])

    fig = plt.figure(facecolor='#b7b7b700')
    axis = fig.add_subplot(1, 1, 1, projection="3d",facecolor='#b7b7b700')
    draw_hsv3d(axis, testimg, pixel_colors)
    fig.savefig(prename+'_HSVspace_map.png', facecolor=fig.get_facecolor(), dpi=200)

    for x, y, xl, yl, xlim, ylim, fname in [
            (h, v, "Hue", "Value", (0, 180), (0, 255), '_HSV_hue-val_map.png'),
            (s, v, "Saturation", "Value", (0, 255), (0, 255), '_HSV_sat-val_map.png'),
            (h, s, "Hue", "Saturation", (0, 180), (0, 255), '_HSV_hue-sat_map.png')]:
        fig = plt.figure(facecolor='#ffffff00')
        axis = fig.add_subplot(111, facecolor='#ffffff00', xlim=xlim, ylim=ylim)
        draw_scatter2d(axis, x.flatten(), y.flatten(), pixel_colors, psize, xl, yl)
        fig.savefig(prename+fname, facecolor=fig.get_facecolor(), dpi=200)

def labmap(testimg):
    pixel_colors = normPixelColors(testimg)
    fig = plt.figure(figsize=(6, 6),facecolor='#b7b7b700')
    axis = fig.add_axes((0.12, 0.09, 0.8, 0.8), facecolor='#b7b7b700',xlim=(-128, 128),ylim=(-128, 128))
    draw_labcircle(axis, testimg, pixel_colors)
    plt.savefig(prename+'_Lab-circle_map.png', facecolor=fig.get_facecolor(), dpi=200)
    
def kmeanmap(testimg,colnum):
    # 画像で使用されている色一覧。(W * H, 3) の numpy 配列。１次元化する。
    colors = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    # cv2.kmeans に渡すデータは float 型である必要があるため、キャストする。
    colors = colors.astype(np.float32)


    # k平均法でクラスタリングする。

    # クラスタ数
    K = colnum

    # 最大反復回数: 10、移動量の閾値: 1.0
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0

    ret, label, center = cv2.kmeans(
        colors, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    print(f"ret: {ret}, label: {label.shape}, center: {center.shape}")
    # ret: 127443.79220199585, label: (48380, 1), center: (8, 3)

    # 各クラスタに属するサンプル数を計算する。
    _, counts = np.unique(label, axis=0, return_counts=True)

    # matplotlib で棒グラフを作成する。
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3))
    fig.subplots_adjust(wspace=0.5)
    # print(center.astype(np.uint8))
    # print(print(center.astype(np.uint8).reshape(1,center.shape[0],center.shape[1])))
    # print((center.reshape(2)).shape)
    # print(testimg.shape)
    # print((testimg.reshape(-1, 3)).shape)
    rgbCenter=cv2.cvtColor(center.astype(np.uint8).reshape(1,center.shape[0],center.shape[1]), cv2.COLOR_LAB2RGB).reshape(-1,3)


    # matplotlib の引数の仕様上、[0, 1] にして、(R, G, B) の順番にする。
    bar_color = [(r / 255, g / 255, b / 255) for r,g,b in rgbCenter]
    bar_text = [f"({r}, {g}, {b})" for r,g,b in rgbCenter]

    # 画像を表示する。
    ax1.imshow(testimg)
    ax1.set_axis_off()

    # ヒストグラムを表示する。
    ax2.barh(np.arange(K), counts, color=bar_color, tick_label=bar_text)
    # plt.show()
    # 各画素を k平均法の結果に置き換える。
    dst = center[label.ravel()].reshape(testimg.shape)
    dst = dst.astype(np.uint8)
    print(label.ravel())

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(dst, cv2.COLOR_LAB2RGB))
    ax.set_axis_off()

    # plt.show()
    cv2.imwrite(prename + "kmean(lab)-replace_"+str(K)+"colors.png",cv2.cvtColor(dst,cv2.COLOR_LAB2BGR))
    # colorReplace(dst,center.astype(np.uint8))


    barWidth = 1000
    barHeight = 100
    # center_alpha = np.zeros((K,4), dtype = int) 
    # center_alpha[:,0:3] = center.astype(np.uint8)
    # center_alpha[:,3] = counts
    # print ("alpha=",center_alpha)
    print("Center",center)

    RGBtmp=cv2.cvtColor(center.astype(np.uint8).reshape(1,K,3), cv2.COLOR_LAB2RGB)
    # print("RGBtmp:",RGBtmp,RGBtmp.shape)
    HSVtmp=cv2.cvtColor(RGBtmp, cv2.COLOR_RGB2HSV).reshape(-1,3)
    # print("hsvTmp:",HSVtmp,HSVtmp.shape)
    newOrder=np.argsort(HSVtmp[:,0])
    # print("neworder:",newOrder)
    strset=["Hue","Saturation","Value","Area"]
    
    #16進数に変換した配列を作る。
    RGBtmp16str=[]
    for color in RGBtmp.reshape(-1,3):
        RGBtmp16str.append("#{:02x}{:02x}{:02x}".format(color[0],color[1],color[2]))

    #CSVファイルに記録
    with open( prename + "kmean(lab)-replace_"+str(K)+"colors.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["RGB","Hue","Saturation","Value","Area"])
        for i in range(len(RGBtmp16str)):
            writer.writerow([RGBtmp16str[i],HSVtmp[i,0],HSVtmp[i,1],HSVtmp[i,2], counts[i]])
            # writer.writerow([RGBtmp16str[i]])

    #ColorBarのレンダリング
    newimg = np.array([]) 
    scalenum = barWidth  /  sum(counts)
    for j in range(4):
        #h,s,v,Areaの順で順序番号を得る
        newOrder=np.argsort(HSVtmp[:,j]) if j != 3 else np.argsort(counts)
        for i in newOrder:
            vol=int(counts[i] * scalenum)
            # print("kemeanLoop_",i,vol,counts[i])
            newimg = np.full((barHeight,vol,3),center.astype(np.uint8)[i]) if i == newOrder[0] else np.hstack((newimg,np.full((barHeight,vol,3),center.astype(np.uint8)[i])))
        # cv2_imshow(cv2.cvtColor(newimg,cv2.COLOR_LAB2BGR))
        cv2.imwrite(prename + "kmean(lab)-color_Bar("+strset[j]+")-"+str(K)+"colors.png",cv2.cvtColor(newimg,cv2.COLOR_LAB2BGR))

    return center.astype(np.uint8), counts

def nearColor(color,LabColors):
    colordelta=[]
    #一度、全ての組み合わせで色差を計算する。
    for basecolor in LabColors:
        deffcolor=basecolor.astype(np.int8)-color.astype(np.int8)
        colordelta.append(np.sqrt(deffcolor[0]**2 + deffcolor[1]**2 + deffcolor[2]**2))
    #色差順で並び替えして、１番色差の低い色を返す。
    return LabColors[np.argsort(colordelta)[0]].flatten()


def scale(color):
    scalewidth=[]
    scalecenter=[]
    for i in range(color.shape[1]):
        maxcolor=np.amax(color[:,i])
        mincolor=np.amin(color[:,i])
        scalewidth.append((maxcolor-mincolor))
        scalecenter.append(maxcolor-(maxcolor-mincolor)/2)
    return scalewidth,scalecenter

def scale6(color):
    scalerange=[]
    for i in range(color.shape[1]):
        maxcolor=np.amax(color[:,i])
        scalerange.append(maxcolor)
    return scalerange

def colorSlide(img,crange):
    imgfloat=img.reshape(-1,3)
    print(imgfloat.shape)
    imgfloat6=trans6lab(colorlab8bit2float(imgfloat))
    drange= scale6(imgfloat6)
    modrange= np.array(crange)/ np.array(drange)
    # modcenter= np.array(ccenter)- np.array(drange)

    print("colorSlide:",drange,crange,modrange)

    # img -= modcenter.astype(np.uint8)
    newimg = imgfloat6 * modrange
    return colorlabfloat28bit(decord3lab(newimg)).reshape(img.shape)

def nearScaleColor(color,LabColors):
    colordelta=[]
    #一度、全ての組み合わせで色差を計算する。
    for basecolor in LabColors:
        deffcolor=basecolor.astype(np.int8)-color.astype(np.int8)#もともとがUint（符号なし）のため、差分の計算にエラーがでる、符号ありの型にキャストする。
        colordelta.append(np.sqrt(deffcolor[0]**2+ deffcolor[1]**2+deffcolor[2]**2))
    #色差順で並び替えして、１番色差の低い色を返す。
    return LabColors[np.argsort(colordelta)[0]].flatten()

'''特定の画像を別の配色に置き換える。
img=変換したい画像（Lab）
LabColor=色(lab)の多次元配列（dtype=uint8）
あまりうまくいかないので、保留'''
def colorReplace(img,LabColors):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            replaceColor=nearColor(img[y,x],LabColors)
            text="color replace"+str(y)+"/"+str(img.shape[0])+":"+str(img[y,x])+">>"+str(replaceColor)
            sys.stdout.write("\r{}".format(text))
            sys.stdout.flush()
            img[y,x]=replaceColor
    cv2.imwrite(prename + "-ColorReplace.png",cv2.cvtColor(img,cv2.COLOR_LAB2BGR))

def quckresize(img,height):
    return cv2.resize(img,(height,int(height * img.shape[0]/img.shape[1])),interpolation = cv2.INTER_NEAREST)

'''シート類の保存: 高解像度PNGで保存し、そのPNGを1ページに埋め込んだラスタPDFも生成する。
(フォント埋め込みPDFはビューアによって日本語が文字化けするため、ラスタ形式で回避)'''
def savefigRaster(fig, basepath, dpi=200):
    fig.savefig(basepath + ".png", facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    PILImage.open(basepath + ".png").convert("RGB").save(basepath + ".pdf", resolution=dpi)

'''分析グラフを1枚の横長シートへ直接描画してPNG/PDF書き出す。
グラフ類は画像の貼り込みではなくシート上へ直接描画するため、軸ラベルも高解像度が保たれる。
raw_img=元画像(BGR)、testimg=縮小済みRGB画像、labCenters/counts=kmeanmapの結果(Lab uint8/面積)'''
def makeSummarySheet(raw_img,testimg,img_title,labCenters,counts):
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'sans-serif']

    fig = plt.figure(figsize=(19.2, 10.8), facecolor='#b7b7b7')

    def underline(x0, x1, y):
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color='black', linewidth=0.8, transform=fig.transFigure))

    pixel_colors = normPixelColors(testimg)
    h, s, v = cv2.split(cv2.cvtColor(testimg, cv2.COLOR_RGB2HSV))
    psize = returnPointSize(h.flatten().shape[0])

    # 左上: 氏名欄
    fig.text(0.03, 0.955, "氏名", fontsize=11)
    underline(0.06, 0.24, 0.952)

    # 左: 元画像 + グレースケール
    ax = fig.add_axes([0.025, 0.55, 0.155, 0.38], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    ax = fig.add_axes([0.19, 0.55, 0.155, 0.38], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), cmap='gray')

    # 左下: 3D散布図 (直接描画)
    ax3 = fig.add_axes([0.005, 0.235, 0.19, 0.30], projection='3d', facecolor='#b7b7b700')
    draw_rgb3d(ax3, testimg, pixel_colors)
    fig.text(0.10, 0.185, "RGB色空間マッピング", fontsize=9, ha='center')
    ax3 = fig.add_axes([0.18, 0.235, 0.19, 0.30], projection='3d', facecolor='#b7b7b700')
    draw_hsv3d(ax3, testimg, pixel_colors)
    fig.text(0.275, 0.185, "HSV色空間マッピング", fontsize=9, ha='center')

    # 中央: HSV 2D散布図 x3 (直接描画)
    fig.text(0.42, 0.945, "HSV空間マッピング", fontsize=13)
    for x, y, xl, yl, xlim, ylim, rect in [
            (h, v, "Hue", "Value", (0, 180), (0, 255), [0.415, 0.745, 0.145, 0.17]),
            (h, s, "Hue", "Saturation", (0, 180), (0, 255), [0.415, 0.49, 0.145, 0.17]),
            (s, v, "Saturation", "Value", (0, 255), (0, 255), [0.415, 0.235, 0.145, 0.17])]:
        axs = fig.add_axes(rect, facecolor='#ffffff00', xlim=xlim, ylim=ylim)
        draw_scatter2d(axs, x.flatten(), y.flatten(), pixel_colors, psize, xl, yl)

    # 右: マンセル色相環・全画素マッピング (Labサークルの代替。Labサークル画像自体は個別ファイルとして出力済み)
    pangles, pchromas = munsellPixelData(testimg)
    rmax = max(14.0, float(np.max(pchromas)) + 1)
    fig.text(0.66, 0.945, "マンセル色相環マッピング (Hue / Chroma)", fontsize=13)
    axp = fig.add_axes([0.62, 0.28, 0.34, 0.62], projection='polar', facecolor='#ffffff00')
    draw_munsell_grid(axp, rmax)
    axp.scatter(np.radians(pangles), pchromas, facecolors=pixel_colors, marker=".", s=psize*4, zorder=3)

    # 下段: K平均法カラーバー
    K = labCenters.shape[0]
    fig.text(0.975, 0.225, "K平均法による代表色の抽出と面積比", fontsize=10, ha='right')
    bar_y = [0.145, 0.085, 0.025]
    for label, y in zip(["Hue", "Saturation", "Value"], bar_y):
        fig.text(0.345, y + 0.052, label, fontsize=9)
        bar = plt.imread(prename + "kmean(lab)-color_Bar("+label+")-"+str(K)+"colors.png")
        ax = fig.add_axes([0.345, y, 0.63, 0.05])
        ax.set_axis_off()
        ax.imshow(bar, aspect='auto')

    # 左下: タイトル・作者欄
    fig.text(0.03, 0.095, "タイトル", fontsize=11)
    fig.text(0.095, 0.095, img_title, fontsize=11)
    underline(0.09, 0.30, 0.09)
    fig.text(0.03, 0.04, "作者", fontsize=11)
    underline(0.09, 0.30, 0.035)

    savefigRaster(fig, prename+"_summary")

'''sRGB(0-255) → マンセル記号 への変換。
変換理論 (ASTM D1535 方式):
  1. sRGB を逆ガンマ補正して線形RGBへ戻し、3x3行列で CIE XYZ (D65光源基準) に変換
  2. マンセル表色系の測定基準である C光源 へ色順応変換 (CAT02)
  3. XYZ → xyY (色度座標 x,y + 輝度 Y) に変換
  4. Munsell Renotation Data (1943年の視覚実験による xyY⇔マンセル値の対応表) を
     ASTM D1535 に基づき補間し、xyY からマンセル値 (色相 H, 明度 V, 彩度 C) を逆算
戻り値: (マンセル記号文字列, 色相角度[deg] 無彩色はNone, 明度V, 彩度C)'''
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
        #Renotation Data の範囲外(極端に暗い色など)は無彩色 N として明度のみ記録
        V = float(colour.notation.munsell.munsell_value_ASTMD1535(xyY[2] * 100))
        return "N {:.1f}".format(V), None, V, 0.0
    spec = colour.notation.munsell.munsell_colour_to_munsell_specification(notation)
    if np.isnan(spec[0]):  # 無彩色 (N表記)
        return notation, None, float(spec[1]), 0.0
    #色相環上の角度: ASTM hue (0-100の通し色相番号) を 3.6倍して等間隔配置 (標準的な色相環図と同じ)
    hue_angle = float(colour.notation.munsell.hue_to_ASTM_hue([spec[0], spec[3]])) * 3.6
    return notation, hue_angle, float(spec[1]), float(spec[2])

'''kmeansで得たLab代表色(uint8)と面積カウントから、マンセル変換済みの色エントリ一覧を作る。
戻り値: (エントリのリスト, 総ピクセル数)。無彩色を先頭、有彩色は色相角順。'''
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

'''マンセル色相環の下地(色相ラベル一周とグリッド)を polar axes に描画する'''
def draw_munsell_grid(axp, rmax):
    #色相環一周ぶん(10色相 x 2.5/5/7.5/10刻み = 40区分)の色相記号を9°間隔で等配置
    hue_angles, hue_labels = [], []
    for fam in ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']:
        for step in ['2.5', '5.0', '7.5', '10.0']:
            spec = colour.notation.munsell.munsell_colour_to_munsell_specification(step + fam + " 5.0/10.0")
            hue_angles.append(float(colour.notation.munsell.hue_to_ASTM_hue([spec[0], spec[3]])) * 3.6)
            hue_labels.append(step.rstrip('0').rstrip('.') + fam)
    axp.set_thetagrids(hue_angles, hue_labels, fontsize=8)
    #基準色相(5R,5YRなど)のラベルは強調表示
    for lbl in axp.get_xticklabels():
        if lbl.get_text().startswith('5'):
            lbl.set_fontsize(11)
            lbl.set_fontweight('bold')
    axp.set_rlim(0, rmax)
    axp.set_rgrids(np.arange(4, rmax + 0.1, 4), fontsize=8, color='#00000088')
    axp.grid(color='#00000022', linewidth=0.4)

'''マンセル色相環 (角度=色相 Hue, 半径=彩度 Chroma, 点サイズ=面積比)'''
def draw_munsell_circle(axp, entries, total, rmax):
    draw_munsell_grid(axp, rmax)
    angles = [0 if e["angle"] is None else e["angle"] for e in entries]
    axp.scatter(np.radians(angles), [e["chroma"] for e in entries],
                c=[e["rgb01"] for e in entries],
                s=[120 + 1200 * e["count"] / total for e in entries],
                edgecolors='#00000055', zorder=3)

_munsell_cache = {}
'''全画素をマンセル(色相角, 彩度)へ変換する。
厳密な逆算(rgb2munsell)は1色あたり数十msかかるため、Labを量子化して固有色ごとに
変換し(結果はキャッシュ)、固有色数が max_unique 以下になる最小の量子化幅を自動選択する。
戻り値: (色相角の配列[deg], 彩度の配列) いずれも画素数ぶん(row-major順)'''
def munsellPixelData(testimg, max_unique=600):
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

'''明度・彩度図 (x=Chroma, y=Value 0-10, グリッド1刻み)'''
def draw_value_chroma(axv, entries, total, rmax):
    axv.set_xlim(0, rmax)
    axv.set_ylim(0, 10)
    #グリッドは1刻みで描画(彩度軸のラベルのみ2刻みで間引き)
    axv.set_xticks(np.arange(0, rmax + 0.1, 2))
    axv.set_xticks(np.arange(0, rmax + 0.1, 1), minor=True)
    axv.set_yticks(np.arange(0, 10.1, 1))
    axv.grid(which='both', color='#00000022')
    #グラフ端の色でもマーカーが半円にクリップされないよう clip_on=False
    axv.scatter([e["chroma"] for e in entries], [e["value"] for e in entries],
                c=[e["rgb01"] for e in entries],
                s=[120 + 1200 * e["count"] / total for e in entries],
                edgecolors='#00000055', zorder=3, clip_on=False)
    axv.set_xlabel("Chroma (彩度)")
    axv.set_ylabel("Value (明度)")

'''サマリーシート2枚目: 元画像 + K平均法による代表色(10色以下)のカラーバーと
マンセル色相環(Hue-Chroma)・明度彩度図(Value-Chroma)へのマッピングをPDF/PNGで書き出す。'''
def makeMunsellSheet(raw_img,img_title,K=10):
    plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'sans-serif']

    #K平均法で代表色を抽出 (Lab空間でクラスタリング)
    testimg = cv2.cvtColor(quckresize(raw_img,100), cv2.COLOR_BGR2RGB)
    data = cv2.cvtColor(testimg, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
    ret, label, center = cv2.kmeans(
        data, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(label, return_counts=True)
    entries, total = munsellEntriesFromCenters(center.astype(np.uint8), counts)

    #マンセル値をCSVにも記録
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

    #左上: 氏名欄 / 左: 元画像 / 左下: タイトル・作者欄
    fig.text(0.03, 0.955, "氏名", fontsize=11)
    underline(0.06, 0.24, 0.952)
    ax = fig.add_axes([0.025, 0.30, 0.30, 0.60], anchor='C')
    ax.set_axis_off()
    ax.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    fig.text(0.03, 0.095, "タイトル", fontsize=11)
    fig.text(0.095, 0.095, img_title, fontsize=11)
    underline(0.09, 0.30, 0.09)
    fig.text(0.03, 0.04, "作者", fontsize=11)
    underline(0.09, 0.30, 0.035)

    rmax = max(14.0, max(e["chroma"] for e in entries) + 2)

    #右上1: マンセル色相環 (角度=色相 Hue, 半径=彩度 Chroma)
    fig.text(0.42, 0.945, "マンセル色相環マッピング (Hue / Chroma)", fontsize=13)
    axp = fig.add_axes([0.36, 0.36, 0.30, 0.54], projection='polar', facecolor='#ffffff00')
    draw_munsell_circle(axp, entries, total, rmax)

    #右上2: 明度・彩度図
    fig.text(0.75, 0.945, "明度・彩度図 (Value / Chroma)", fontsize=13)
    axv = fig.add_axes([0.735, 0.40, 0.23, 0.48], facecolor='#ffffff00')
    draw_value_chroma(axv, entries, total, rmax)

    #下段: 代表色カラーバー (幅=面積比、下にマンセル記号)
    fig.text(0.36, 0.30, "K平均法による代表" + str(K) + "色とマンセル値 (面積比)", fontsize=11)
    axb = fig.add_axes([0.36, 0.16, 0.61, 0.10])
    axb.set_axis_off()
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    x = 0.0
    narrow_i = 0
    #カラーバーは面積比の大きい順(左→右)に並べる
    for e in sorted(entries, key=lambda e: -e["count"]):
        w = e["count"] / total
        axb.add_patch(plt.Rectangle((x, 0), w, 1, facecolor=e["rgb01"], edgecolor='none'))
        #面積比%はセグメント内に表示(明度で文字色を切替、狭すぎる場合は省略)
        if w >= 0.03:
            txtcol = 'white' if e["value"] < 5.5 else 'black'
            axb.text(x + w / 2, 0.5, "{:.0f}%".format(w * 100), ha='center', va='center',
                     fontsize=8, color=txtcol)
        #狭いセグメントが連続するとラベルが重なるため、上下2段に振り分ける
        if w < 0.05:
            ylabel = -0.10 if narrow_i % 2 == 0 else -0.75
            narrow_i += 1
        else:
            ylabel = -0.10
            narrow_i = 0
        axb.text(x + w / 2, ylabel, e["notation"], rotation=40, ha='right', va='top',
                 rotation_mode='anchor', fontsize=9, clip_on=False)
        x += w

    #脚注: 変換理論の説明
    fig.text(0.36, 0.03,
             "RGB→マンセル変換: sRGB →(逆ガンマ補正)→ CIE XYZ (D65) →(色順応変換 CAT02: D65→C光源)→ xyY\n"
             "→ Munsell Renotation Data (ASTM D1535) の補間により 色相H・明度V・彩度C を算出",
             fontsize=9, va='bottom')

    savefigRaster(fig, prename+"_summary2")

def imagesetProcess(file_list,B_img):
    global prename
    for file_path in file_list:
        img_path = file_path
        raw_img = cv2.imread(img_path) 
        prename = img_path.split(".")[0]
        new_path = prename+"color-analysis"
        if os.path.isdir(new_path)==False:
            os.makedirs(new_path)
        prename = new_path + "/"+prename
        cv2.imwrite(prename+"_gray.png", cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY))

        #画像圧縮。分析するにあたり解像度が低すぎる場合は変更してください。元画像がもともと低解像の場合は、コメントアウトして下さい。
        reSizeHSize = 100
        testimg = quckresize(raw_img,reSizeHSize)
        testimg = cv2.cvtColor(testimg, cv2.COLOR_BGR2RGB)

        # pixel_colors = testimg.reshape((np.shape(testimg)[0]*np.shape(testimg)[1], 3))
        rgbmap(testimg)
        hsvmap(testimg)
        labmap(testimg)

        #input RGB return LAB
        newColorSet, kcounts = kmeanmap(testimg,20)
        #生成した分析画像を1枚のシートにまとめてPDF/PNG出力
        makeSummarySheet(raw_img, testimg, os.path.basename(img_path), newColorSet, kcounts)
        #サマリーシート2枚目: 代表10色のマンセル値マッピング
        makeMunsellSheet(raw_img, os.path.basename(img_path), 10)
        plt.close('all')
        #色置き換え対象の画像(B_img)が無い場合は分析のみで終了。
        if B_img is None:
            continue
        #LabのカラーセットをLab+-6chにセパレートして分布レンジを記録した色のセット一式を作成。
        crange=scale6(trans6lab(colorlab8bit2float(newColorSet)))
        #色の置き換えの前段階として、置き換え対象の画像の色分布域値を合わせる。Lab>Lab
        modimg=colorSlide(cv2.cvtColor(B_img, cv2.COLOR_BGR2LAB),crange)
        # cv2.imwrite(prename+"_test.png",cv2.cvtColor(modimg,cv2.COLOR_LAB2BGR))
        #色差を基準にした色の置き換え　input LAB,LAB
        colorReplace(modimg,newColorSet)

if __name__ == '__main__':

    file_list = glob.glob("*.png")
    file_list.extend(glob.glob("*.PNG"))
    file_list.extend(glob.glob("*.jpg"))
    file_list.extend(glob.glob("*.jpeg"))
    file_list.extend(glob.glob("*.JPG"))

    #抽出した配色を用いて対象画像を塗り替えする。ここでその対象画像を指定しておく。
    #ファイルが無い場合は色置き換え工程をスキップして分析のみ行う。
    B_img_raw = cv2.imread('class/woman-with-the-hat.png')
    B_img = quckresize(B_img_raw,400) if B_img_raw is not None else None
    if B_img is None:
        print("[info] class/woman-with-the-hat.png が見つからないため、色置き換え工程はスキップします（分析のみ実行）。")
    #メインプロセス
    imagesetProcess(file_list,B_img)