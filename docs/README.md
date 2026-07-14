# 色分析 Color Analysis (Web版)

画像の配色を分析するWebアプリです。RGB / HSV / CIE L\*a\*b\* の色空間マッピング、
K平均法による代表色抽出、マンセル表色系(ASTM D1535)へのマッピングを行い、
サマリーシート(PDF/PNG)を含む分析画像一式をZIPでダウンロードできます。

処理はすべて **ブラウザ内(Pyodide = WebAssembly上のPython)** で実行されます。
サーバーは不要で、アップロードした画像が外部に送信されることはありません。

## GitHub Pages での公開手順

このフォルダ (`docs/`) はリポジトリごと push した上で、GitHub Pages の配信元に指定する。

1. リポジトリを GitHub に push する
2. リポジトリの **Settings → Pages** を開き、
   Source: **Deploy from a branch** / Branch: **main** / フォルダ: **/docs** を選んで Save
3. 数分後に `https://<ユーザー名>.github.io/<リポジトリ名>/` で公開される

## ファイル構成

| ファイル | 役割 |
|---|---|
| `index.html` | UI と Pyodide の読み込み・実行を行うページ本体 |
| `analysis.py` | 分析ロジック (ColorAnalysis2.py のブラウザ向け移植版) |
| `sample.png` | 「デモ画像で試す」用のサンプル画像 |

## 制限・注意

- 初回アクセス時はライブラリ(数十MB)のダウンロードに1〜2分かかります(2回目以降はキャッシュされます)
- 1枚あたりの分析には数十秒〜数分かかります(WebAssembly上で実行されるため、ネイティブ実行より低速です)
- デスクトップ版にある実験的な色置き換え工程(colorReplace)は含まれていません

## RGB→マンセル変換の理論

sRGB →(逆ガンマ補正)→ CIE XYZ (D65光源) →(色順応変換 CAT02: D65→C光源)→ xyY →
Munsell Renotation Data (1943年の視覚実験による対応表) を ASTM D1535 に基づき補間して
色相H・明度V・彩度C を算出しています。実装には
[colour-science](https://www.colour-science.org/) ライブラリを使用しています。
