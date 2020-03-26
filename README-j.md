# ギブスサンプリングを用いた非負値行列分解の学習係数の数値計算

本リポジトリは，非負値行列分解の実対数閾値（またの名を学習係数）（学習率ではない）を，ギブスサンプリングによる事後分布から計算する`Julia 1.3.0`プログラムについてのものです．
このリポジトリにあるソースコードは林による非負値行列分解の変分近似誤差の理論研究[1]における数値実験に使われました.

## 実行環境

ここでは，筆者が数値実験に用いた環境を示します．

### ハードウェアとOS

* CPU: Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
* RAM: 16.0 GB
* OS: Windows 10 Home 64bit

### ソフトウェア

* Julia言語 version 1.3.0 (より新しいバージョンでも動くかもしれませんが試していません).

また，下記のパッケージを使いました．
```
  "PDMats"           => v"0.9.11"
  "Distributions"    => v"0.21.12"
  "StatsPlots"       => v"0.13.0"
  "IJulia"           => v"1.20.2"
  "ProgressMeter"    => v"1.2.0"
  "Gadfly"           => v"1.0.1"
  "StatsFuns"        => v"0.9.3"
  "SpecialFunctions" => v"0.9.0"
```

## 何をするプログラムか

ポアソン＝ガンマモデルで構築したベイズ非負値行列分解の事後分布をギブスサンプリングによって実現した際のベイズ汎化誤差とWAICを数値計算し，
そのデータセットの出方に関する平均から実対数閾値（学習係数）を計算します．

## 内容

* `README.md`: このファイルの英語版.
* `README-j.md`: このファイル.
* `Julia_calculate_RLCT_of_NMF_by_GS_NEUNET.ipynb`: 実験コードのJupyter notebookファイル.
* `Julia_calculate_RLCT_of_NMF_by_GS_NEUNET.jl`: 上記のipynbファイルから出力したJuliaコードファイル.
* `image/PhaseTrans-withdesc.png`: 後述される非負値行列分解の相図.

## 理論的背景

理論の結果[1]より，実対数閾値は事前分布（ガンマ分布）のハイパーパラメータに依存し，変分近似誤差の下限がハイパーパラメータによって劇的に変化することがわかります．

![image/PhaseTrans-withdesc.png](image/PhaseTrans-withdesc.png "非負値行列分解のハイパーパラメータに関する相図")

この現象は物理学からの借用で「相転移」と呼ばれ，臨界線は「相転移線」といいます．
変分ベイズ非負値行列分解の厳密な相転移構造が幸島と渡辺により解明されています[2]（上の図の青い直線です）が，
ベイズ非負値行列分解の場合はわかっていませんでした．林の研究[1]では実対数閾値の上界と変分近似誤差の下限を導出し，そこにある相転移構造を理論的に発見しました（上の図の赤い破線です）．
詳細はJupyter notebookファイル`Julia_calculate_RLCT_of_NMF_by_GS_NEUNET.ipynb`のマークダウンセルか文献[1]を参考ください．
本コードは文献[1]における主定理の数値的挙動を確認するための実験に用いられました．上の図の黒い点が論文中の実験に用いたハイパーパラメータです．

## 引用文献

1. Naoki Hayashi. "Variational Approximation Error in Non-negative Matrix Factorization", Neural Networks, Volume 126, June 2020, pp.65-75. [doi: 10.1016/j.neunet.2020.03.009](https://doi.org/10.1016/j.neunet.2020.03.009). (2019/6/18 submitted. 2020/3/9 accepted. 2020/3/20 published on web). The arXiv version is [here, arXiv: 1809.02963](https://arxiv.org/abs/1809.02963).

2. Masahiro Kohjima and Sumio Watanabe. "Phase Transition Structure of Variational Bayesian Nonnegative Matrix Factorization", In: Lintas A., Rovetta S., Verschure P., Villa A. (eds) Artificial Neural Networks and Machine Learning – ICANN 2017. ICANN 2017. Lecture Notes in Computer Science, vol 10614. Springer, Cham. [doi: 10.1007/978-3-319-68612-7_17](https://https://doi.org/10.1007/978-3-319-68612-7_17)


なお，文献[1]の日本語版としてIBIS2018におけるテクニカルレポート（T-08）があります．[IBIS2018のHP](http://ibisml.org/ibis2018/technical/)や[電子情報通信学会のHP](https://www.ieice.org/ken/paper/20181105d1Hq/)からアブストラクトが読めます．
また，文献[2]はICANN2017のBest Paper Award受賞論文であり，[東工大ニュース](https://educ.titech.ac.jp/is/news/2017_10/054783.html)にも掲載されています．
