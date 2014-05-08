## 独立成分分析（ICA）について
独立成分分析とは，複数の観測信号をもとに複数の生成信号を復元するための手法であり多変量解析手法の1つである．独立成分分析の良く知られた問題としてカクテルパーティー問題がある．
> カクテルパーティでは多くの人同時にそれぞれ話しをしている．ところが，これらの多くの声の中から，自分が注目している声を人間は聞き分けることができる．この現象をカクテルパーティ効果という．この効果を独立成分分析によって実現する研究が行われている．
引用: [カクテルパーティー効果](http://ibisforest.org/index.php?%E3%82%AB%E3%82%AF%E3%83%86%E3%83%AB%E3%83%91%E3%83%BC%E3%83%86%E3%82%A3%E5%8A%B9%E6%9E%9C)

すなわち独立成分分析では，m個の観測信号
$x(t) = [x_1(t), x_2(t), \cdots, x_m(t)]'$
が観測されたときに，その背景にあるn個の生成信号（情報源）
$s(t) = [s_1(t), s_2(t), \cdots, s_n(t)]'$
を復元することを目指す．

独立成分分析では，このとき次の2つの仮定をおいてこの問題を解く

1. 観測信号x(t)は，n個の生成信号s(t)の線形結合で表される．
すなわち，$x(t) = As(t)$なるm×nの行列Aが存在する．
2. 情報源の各信号$s_1(t), s_2(t), \cdots, s_n(t)$はそれぞれ独立であるとする．

この仮定に従うと，$s(t) = A^{-1}x(t)$の変換によって
復元したn個の成分$s(t)$がそれぞれ独立になるように$A^{-1}$を求めれば良い．このように，観測信号から独立な成分を抽出して情報源の予測値とみなすのが独立成分分析である．

## 参考サイト ##
独立成分分析: [http://ibisforest.org/index.php?%E7%8B%AC%E7%AB%8B%E6%88%90%E5%88%86%E5%88%86%E6%9E%90](http://ibisforest.org/index.php?%E7%8B%AC%E7%AB%8B%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)  
blind source separationのデモ: [http://research.ics.aalto.fi/ica/cocktail/cocktail_en.cgi](http://research.ics.aalto.fi/ica/cocktail/cocktail_en.cgi)  
sklearnのFastICAのリファレンス: [http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html](http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)