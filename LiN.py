import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from lingam import DirectLiNGAM
from lingam.utils import make_dot
from sklearn.preprocessing import StandardScaler
from scipy import stats

# データの読み込み
file_path = 'BMHXRM:BEAM:SIGMAYatIP_202412060000-202412080000.csv'  # データファイルのパス
df = pd.read_csv(file_path)

# PV名に含まれるドットとコロンをアンダースコアに置き換え
df.columns = [col.replace('.', '_').replace(':', '_') for col in df.columns]

# 不要な列を削除
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns=['BMHXRM_BEAM_SIGMAYatIP'])

# NaN を含む行を削除（もし必要なら）
df = df.dropna()

print(df)
print(df.columns)

# データの標準化
scaler = StandardScaler()
norm_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# LiNGAMの実行（事前知識行列なし）
model_lingam = DirectLiNGAM()
model_lingam.fit(norm_df)

# 結果の出力
print("Causal Order:", model_lingam.causal_order_)
print("Adjacency Matrix:")
print(model_lingam.adjacency_matrix_)

# 因果グラフの描画
dot = make_dot(model_lingam.adjacency_matrix_, labels=df.columns.tolist())  # .tolist() でリストに変換

# PDFとして保存
dot.render('dag')

# PNGとして保存
dot.format = 'png'
dot.render('dag')

# 誤差項の独立性の検定
p_values = model_lingam.get_error_independence_p_values(norm_df)
print("P-values of error independence:")
print(p_values)

# 最初の1000行を使用
df_sample = df.sample(n=min(1000, len(df)), random_state=42)  # サンプル数を1000に固定
norm_df = scaler.fit_transform(df_sample)
model_lingam.fit(norm_df)

# 残差の正規性検定（Shapiro-Wilk検定）
X = pd.DataFrame(norm_df, columns=df.columns)
X_c = X.values - X.values.mean(axis=0)
E = X_c - (model_lingam.adjacency_matrix_ @ X_c.T).T

# 残差の正規性検定と可視化
fig, axes = plt.subplots(1, len(df.columns), figsize=(16, 4))
fig.suptitle('Shapiro-Wilk tests for residuals')

for i, (col, ax) in enumerate(zip(df.columns, axes)):
    ei = E[:, i]
    p_value = stats.shapiro(ei)[1]
    ax.set_title(f'Residual of {col}\n(p-value: {p_value:.4f})')
    ax.set_ylabel('Density')
    ax.hist(ei, bins=30, density=True, alpha=0.3, label='Residual Histogram')
    xlim = ax.get_xlim()
    x_pdf = np.linspace(xlim[0], xlim[1], 1000)
    ax.plot(x_pdf, stats.norm.pdf(x_pdf, loc=ei.mean(), scale=ei.std()), c='r', label='Normal PDF')
    ax.legend()

fig.show()