# Regularized NLCG Inversion

## 日本語

## 概要

このリポジトリは、正則化付き非線形共役勾配法を用いた逆解析コードのPython実装です。
元のJupyter Notebookで作成していた1次元インバージョン計算のうち、順解析、真モデルの作成、グラフ描画、Notebook上での確認用セルなどを取り除き、逆解析の計算部分だけを再利用しやすい形で整理しています。

このコードでは、観測データに対して任意の順解析モデルを外部から与えることで、モデルパラメータを推定します。
順解析関数はコード内部に固定せず、`forward_model(model, x_data)` という形式でユーザーが自由に定義して渡せるようにしています。
そのため、電気探査、物理モデルのパラメータ推定、またはその他の非線形逆問題に対して、順解析部分を差し替えることで利用できます。

モデルパラメータには上限値と下限値を設定でき、内部では有界なパラメータを無制約空間へ変換して更新します。
これにより、物理的に意味のある範囲を保ちながら、逆解析を安定して進めることを目的としています。

また、残差は対数空間で評価されます。
観測値と計算値の比に近い形で誤差を扱うため、値のスケールが大きく異なるデータを扱う場合にも有効です。
ただし、対数を用いるため、観測データおよび計算データは正の値であることが前提になります。

## 主な機能

* 正則化付き非線形共役勾配法による逆解析
* 上限値・下限値を持つモデルパラメータの取り扱い
* 有界パラメータから無制約空間への変換
* 無制約空間から元のモデル空間への逆変換
* 対数空間での残差評価
* 中央差分によるヤコビアン行列の数値近似
* RMSPEによる誤差評価
* 反復ごとの誤差履歴、正則化パラメータ履歴、モデル履歴の保存
* 任意の順解析関数を外部から渡せる汎用的な構成
* Notebook依存の処理を除いた、再利用しやすいPythonモジュール

## ファイル構成

```text
regularized_nlcg_inversion.py
```

このファイルには、正則化付き非線形共役勾配法による逆解析に必要な関数がまとめられています。

主な関数は以下の通りです。

```python
modellog(linear_model, model_min, model_max)
```

有界なモデルパラメータを、更新しやすい無制約空間へ変換します。
モデルパラメータが上限値・下限値の範囲内に収まるようにしながら、内部計算では制約のない変数として扱えるようにします。

```python
return_log(log_model, model_min, model_max)
```

無制約空間で表されたモデルパラメータを、元の上限値・下限値を持つモデル空間へ戻します。
逆解析の更新は無制約空間で行い、順解析関数に渡すときには元の物理的なスケールのモデルへ戻します。

```python
rmspe(calculated_data, observed_data)
```

計算値と観測値の誤差を、Root Mean Squared Percentage Errorとして評価します。
反復計算中の収束状況を確認するために使用します。

```python
finite_difference_log_jacobian(...)
```

対数残差に対するヤコビアン行列を、中央差分によって数値的に近似します。
順解析関数の解析的な微分を用意しなくても、任意の非線形モデルに対して逆解析を行えるようにするための関数です。

```python
regularized_nlcg_inversion(...)
```

正則化付き非線形共役勾配法を用いて、非線形逆問題を解くメインの関数です。
観測データ、初期モデル、モデルパラメータの上下限、順解析関数などを入力として、最終的に推定されたモデルパラメータを返します。

## 必要な環境

このコードを実行するには、以下の環境が必要です。

```text
Python 3
NumPy
```

NumPyがインストールされていない場合は、以下のコマンドでインストールできます。

```bash
pip install numpy
```

## 使い方

このコードでは、ユーザーが任意の順解析関数を定義し、その関数を逆解析ソルバーに渡します。
順解析関数は、以下の形式で定義します。

```python
calculated_data = forward_model(model, x_data)
```

ここで、`model` は推定したいモデルパラメータの配列、`x_data` は順解析に必要な入力データです。
`forward_model` は、与えられたモデルパラメータと入力データから、観測データに対応する計算値を返す必要があります。

例えば、比抵抗と層厚を同時に推定したい場合は、それらを1つのモデルベクトルとして結合し、`forward_model` の内部で必要な形に分割して使用します。

## 使用例

以下は、簡単な指数関数モデルを用いた使用例です。

```python
import numpy as np
from regularized_nlcg_inversion import regularized_nlcg_inversion

def forward_model(model, x_data):
    a, b = model
    return a * np.exp(-b * x_data)

x_data = np.linspace(0, 10, 50)
observed_data = 2.5 * np.exp(-0.4 * x_data)

initial_model = np.array([1.0, 0.1])
model_min = np.array([0.1, 0.01])
model_max = np.array([10.0, 2.0])

result = regularized_nlcg_inversion(
    forward_model=forward_model,
    x_data=x_data,
    observed_data=observed_data,
    initial_model=initial_model,
    model_min=model_min,
    model_max=model_max,
    max_iter=1000,
    stop_rmspe=1.0,
)

print(result["model_final"])
print(result["rmspe_final"])
```

## 引数

`regularized_nlcg_inversion` の主な引数は以下の通りです。

```python
forward_model
```

モデルパラメータと入力データから計算値を求める順解析関数です。
この関数を差し替えることで、さまざまな非線形逆問題に対応できます。

```python
x_data
```

順解析関数で使用する入力データです。
測定点、周波数、距離、時間、またはモデル計算に必要な任意の配列を指定します。

```python
observed_data
```

逆解析でフィッティングする観測データです。
残差は対数空間で評価されるため、基本的には正の値を持つデータを想定しています。

```python
initial_model
```

逆解析の初期モデルです。
モデルパラメータの初期値を1次元配列として与えます。

```python
model_min
```

各モデルパラメータの下限値です。
物理的に取り得ない値へ更新されないようにするために使用します。

```python
model_max
```

各モデルパラメータの上限値です。
`model_min` と合わせて、モデルパラメータの探索範囲を設定します。

```python
max_iter
```

最大反復回数です。
収束条件を満たさない場合でも、この回数に達すると計算を終了します。

```python
stop_rmspe
```

RMSPEに基づく停止条件です。
RMSPEがこの値を下回ると、反復計算を終了します。

```python
diff_step
```

中央差分でヤコビアンを数値近似するときの差分幅です。
小さすぎると数値誤差の影響を受けやすく、大きすぎると微分近似の精度が低下する場合があります。

```python
q
```

正則化パラメータを反復ごとに減衰させるための係数です。
値が小さいほど、正則化の影響が早く弱くなります。

```python
p
```

モデル更新量にかける減衰係数です。
更新が大きすぎる場合は、この値を小さくすることで安定化できる可能性があります。

```python
alpha_init
```

初期正則化パラメータです。
指定しない場合は、初期計算に基づいて自動的に推定されます。

```python
reference_log_model
```

正則化項で使用する参照モデルです。
指定しない場合は、無制約空間におけるゼロベクトルが参照モデルとして使用されます。

```python
verbose
```

`True` にすると、反復ごとのRMSPE、正則化パラメータ、ステップ長などの進行状況を表示します。

## 返り値

`regularized_nlcg_inversion` は、推定結果を辞書形式で返します。

主な返り値は以下の通りです。

```python
result["model_final"]
```

最終的に推定されたモデルパラメータです。

```python
result["calculated_final"]
```

最終モデルを用いて計算されたデータです。

```python
result["rmspe_hist"]
```

各反復におけるRMSPEの履歴です。

```python
result["alpha_hist"]
```

各反復における正則化パラメータの履歴です。

```python
result["model_hist"]
```

各反復におけるモデルパラメータの履歴です。

```python
result["rmspe_final"]
```

最終反復におけるRMSPEです。

```python
result["elapsed"]
```

計算にかかった時間です。

```python
result["iterations"]
```

実行された反復回数です。

## 正則化付き非線形共役勾配法について

非線形逆問題では、観測データに対してモデルパラメータを推定します。
しかし、観測データにはノイズが含まれることがあり、また逆問題自体が不安定になる場合もあります。
そのため、データへのフィッティングだけを重視すると、推定されたモデルが不自然に大きく変動したり、物理的に解釈しにくい結果になることがあります。

このコードでは、データ残差に加えて正則化項を導入し、モデル更新を安定化させています。
正則化項は、モデルが参照モデルから大きく離れすぎないようにする役割を持ちます。
また、反復計算が進むにつれて正則化パラメータを徐々に小さくすることで、初期段階では安定性を重視し、後半では観測データへの適合を高めるようにしています。

共役勾配法は、勾配方向だけでなく、過去の探索方向も利用しながら解を更新する方法です。
単純な最急降下法よりも効率的に解へ近づけることが期待できます。
本コードでは、非線形問題に対してこの考え方を適用し、反復的にモデルパラメータを更新します。

## このコードで除外しているもの

このファイルは、逆解析ソルバーとして再利用しやすくすることを目的としているため、以下の処理は含めていません。

* 特定の順解析関数
* 真モデルの作成
* 合成観測データの作成
* グラフ描画
* Notebook上での確認用コード
* 特定の探査配置や物理モデルに依存する処理

これらの処理は、解析対象に応じて別ファイルまたはNotebook側で用意してください。

## 注意点

このコードは、任意の非線形順解析モデルに適用できるように汎用的に作成されています。
ただし、実際の解析に使用する場合は、対象とするモデルや観測データの性質に合わせて、初期モデル、モデルパラメータの上下限、正則化パラメータ、反復回数、収束条件、差分ステップ幅などを適切に調整する必要があります。

また、残差は対数空間で評価されるため、観測データと計算データは正の値を持つことが前提です。
ゼロや負の値を含むデータを扱う場合は、前処理や順解析モデルの設計に注意が必要です。

さらに、ヤコビアンは中央差分によって数値的に近似しているため、パラメータ数が多い場合や順解析計算が重い場合には、計算時間が長くなる可能性があります。



---

## English

## Overview

This repository is a Python implementation of inversion code using a regularized nonlinear conjugate-gradient method.
From the original Jupyter Notebook for one-dimensional inversion, forward modeling, true-model creation, plotting, and notebook-specific checking cells have been removed, and only the inversion calculation part has been organized into a reusable form.

In this code, model parameters are estimated by supplying an arbitrary forward model from outside and fitting observed data.
The forward model is not fixed inside the code. Instead, users can freely define and pass a function in the form of `forward_model(model, x_data)`.
Therefore, this code can be applied to electrical exploration, physical model parameter estimation, or other nonlinear inverse problems by replacing the forward modeling part.

Upper and lower bounds can be assigned to the model parameters, and bounded parameters are internally transformed into an unconstrained space before being updated.
This is intended to make the inversion stable while keeping the parameters within physically meaningful ranges.

The residual is evaluated in logarithmic space.
Because the error is treated in a form close to the ratio between observed and calculated values, this approach can be useful when the data have different scales.
However, since logarithms are used, the observed data and calculated data are assumed to have positive values.

## Main Features

* Inversion using a regularized nonlinear conjugate-gradient method
* Handling of model parameters with upper and lower bounds
* Transformation from bounded parameters to an unconstrained space
* Inverse transformation from the unconstrained space to the original model space
* Residual evaluation in logarithmic space
* Numerical approximation of the Jacobian matrix using central differences
* Error evaluation using RMSPE
* Storage of error history, regularization parameter history, and model history for each iteration
* General structure that allows an arbitrary forward model to be passed from outside
* Reusable Python module without notebook-dependent processing

## File Structure

```text
regularized_nlcg_inversion.py
```

This file contains the functions required for inversion using a regularized nonlinear conjugate-gradient method.

The main functions are as follows.

```python
modellog(linear_model, model_min, model_max)
```

This function transforms bounded model parameters into an unconstrained space that is easier to update.
It allows the model parameters to be treated as unconstrained variables internally while keeping them within their upper and lower bounds.

```python
return_log(log_model, model_min, model_max)
```

This function transforms model parameters represented in the unconstrained space back into the original model space with upper and lower bounds.
The inversion update is performed in the unconstrained space, and the model is converted back to the original physical scale when it is passed to the forward model.

```python
rmspe(calculated_data, observed_data)
```

This function evaluates the error between calculated data and observed data as the Root Mean Squared Percentage Error.
It is used to check the convergence during the iterative calculation.

```python
finite_difference_log_jacobian(...)
```

This function numerically approximates the Jacobian matrix of the logarithmic residual using central differences.
It allows inversion to be performed for arbitrary nonlinear models without preparing analytical derivatives of the forward model.

```python
regularized_nlcg_inversion(...)
```

This is the main function that solves a nonlinear inverse problem using a regularized nonlinear conjugate-gradient method.
It takes observed data, an initial model, parameter bounds, and a forward model function as inputs, and returns the finally estimated model parameters.

## Requirements

The following environment is required to run this code.

```text
Python 3
NumPy
```

If NumPy is not installed, it can be installed using the following command.

```bash
pip install numpy
```

## Usage

In this code, the user defines an arbitrary forward model and passes that function to the inversion solver.
The forward model should be defined in the following form.

```python
calculated_data = forward_model(model, x_data)
```

Here, `model` is an array of model parameters to be estimated, and `x_data` is the input data required for the forward calculation.
The `forward_model` must return calculated values corresponding to the observed data from the given model parameters and input data.

For example, when estimating both resistivity and layer thickness, they can be combined into one model vector and split into the required form inside `forward_model`.

## Example

The following is a simple example using an exponential function model.

```python
import numpy as np
from regularized_nlcg_inversion import regularized_nlcg_inversion

def forward_model(model, x_data):
    a, b = model
    return a * np.exp(-b * x_data)

x_data = np.linspace(0, 10, 50)
observed_data = 2.5 * np.exp(-0.4 * x_data)

initial_model = np.array([1.0, 0.1])
model_min = np.array([0.1, 0.01])
model_max = np.array([10.0, 2.0])

result = regularized_nlcg_inversion(
    forward_model=forward_model,
    x_data=x_data,
    observed_data=observed_data,
    initial_model=initial_model,
    model_min=model_min,
    model_max=model_max,
    max_iter=1000,
    stop_rmspe=1.0,
)

print(result["model_final"])
print(result["rmspe_final"])
```

## Arguments

The main arguments of `regularized_nlcg_inversion` are as follows.

```python
forward_model
```

The forward modeling function that calculates predicted data from model parameters and input data.
By replacing this function, the solver can be applied to various nonlinear inverse problems.

```python
x_data
```

The input data used by the forward model.
This can be measurement points, frequencies, distances, time values, or any array required for the model calculation.

```python
observed_data
```

The observed data to be fitted by the inversion.
Since the residual is evaluated in logarithmic space, the data are basically assumed to have positive values.

```python
initial_model
```

The initial model for the inversion.
The initial values of the model parameters are given as a one-dimensional array.

```python
model_min
```

The lower bounds of each model parameter.
These bounds are used to prevent the model from being updated to physically impossible values.

```python
model_max
```

The upper bounds of each model parameter.
Together with `model_min`, this defines the search range of the model parameters.

```python
max_iter
```

The maximum number of iterations.
Even if the convergence condition is not satisfied, the calculation stops when this number is reached.

```python
stop_rmspe
```

The stopping condition based on RMSPE.
When the RMSPE becomes smaller than this value, the iterative calculation stops.

```python
diff_step
```

The finite-difference step width used to numerically approximate the Jacobian by central differences.
If this value is too small, numerical errors may become significant, and if it is too large, the accuracy of the derivative approximation may decrease.

```python
q
```

The factor used to decay the regularization parameter at each iteration.
The smaller this value is, the faster the effect of regularization becomes weaker.

```python
p
```

The damping factor applied to the model update.
If the update is too large, decreasing this value may help stabilize the calculation.

```python
alpha_init
```

The initial regularization parameter.
If it is not specified, it is automatically estimated based on the initial calculation.

```python
reference_log_model
```

The reference model used in the regularization term.
If it is not specified, a zero vector in the unconstrained space is used as the reference model.

```python
verbose
```

When this is set to `True`, the progress of each iteration, such as RMSPE, the regularization parameter, and the step length, is displayed.

## Return Values

`regularized_nlcg_inversion` returns the estimation results as a dictionary.

The main return values are as follows.

```python
result["model_final"]
```

The final estimated model parameters.

```python
result["calculated_final"]
```

The calculated data obtained using the final model.

```python
result["rmspe_hist"]
```

The history of RMSPE values for each iteration.

```python
result["alpha_hist"]
```

The history of the regularization parameter for each iteration.

```python
result["model_hist"]
```

The history of model parameters for each iteration.

```python
result["rmspe_final"]
```

The RMSPE value at the final iteration.

```python
result["elapsed"]
```

The elapsed computation time.

```python
result["iterations"]
```

The number of executed iterations.

## About the Regularized Nonlinear Conjugate-Gradient Method

In a nonlinear inverse problem, model parameters are estimated from observed data.
However, observed data may contain noise, and the inverse problem itself may be unstable.
Therefore, if only data fitting is emphasized, the estimated model may vary unnaturally or become difficult to interpret physically.

In this code, a regularization term is introduced in addition to the data residual to stabilize the model update.
The regularization term plays the role of preventing the model from moving too far away from the reference model.
Also, by gradually reducing the regularization parameter as the iterative calculation proceeds, the early stage emphasizes stability, while the later stage improves the fit to the observed data.

The conjugate-gradient method updates the solution by using not only the gradient direction but also previous search directions.
It is expected to approach the solution more efficiently than a simple steepest-descent method.
In this code, this idea is applied to nonlinear problems, and the model parameters are updated iteratively.

## What Is Excluded From This Code

This file is intended to be reusable as an inversion solver, so the following processes are not included.

* A specific forward modeling function
* True-model creation
* Synthetic observed-data creation
* Plotting
* Notebook-specific checking code
* Processing dependent on a specific survey configuration or physical model

These processes should be prepared in another file or in a Notebook according to the target analysis.

