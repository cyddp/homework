# 第 1 题 代数证明

一元线性回归模型：\(y_i = \hat\beta_0 + \hat\beta_1 x_i + e_i\)残差定义：\(e_i = y_i-\hat y_i = y_i - \hat\beta_0 - \hat\beta_1 x_i\)

最小二乘的目标：\(\min Q(\beta_0,\beta_1)=\sum_{i=1}^n (y_i-\beta_0-\beta_1 x_i)^2\)对\(\beta_0,\beta_1\)求偏导并令导数 = 0，得到正规方程组：

\(\begin{cases}
\displaystyle\frac{\partial Q}{\partial \beta_0} = -2\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)=0\\[4pt]
\displaystyle\frac{\partial Q}{\partial \beta_1} = -2\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)x_i=0
\end{cases}\)1. 证明 \(\boldsymbol{\sum_{i=1}^n e_i =0}\)由第一个方程：

\(\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)=\sum_{i=1}^n e_i = 0\)2. 证明 \(\boldsymbol{\sum_{i=1}^n x_i e_i =0}\)由第二个方程：

\(\sum_{i=1}^n (y_i-\hat\beta_0-\hat\beta_1 x_i)x_i=\sum_{i=1}^n x_i e_i =0\)



# 第 2 题 思考题：新增纯随机无关特征，\(R^2\)上升，调整\(R^2_{adj}\)下降

### 公式回顾

\(R^2 = 1-\frac{RSS}{TSS},\quad R^2_{adj}=1-\frac{RSS/(n-k-1)}{TSS/(n-1)}\)- RSS：残差平方和；TSS总平方和（固定，由y本身决定）
- n样本量，k自变量个数；自由度：残差自由度\(df = n-k-1\)

### 解释

1. **\(R^2\)为什么上升或不变**\(R^2\)只看RSS。往模型加入新自变量（哪怕是纯随机噪声），最小二乘会尽可能利用这个新变量去拟合样本里随机波动，**残差平方和RSS不会变大，只会减小或不变**。TSS不变，RSS↓ → \(R^2=1-RSS/TSS\)上升或不变。


2. **\(R^2_{adj}\)为什么会下降**\(R^2_{adj}\)引入自由度惩罚：分母是残差自由度 \(n-k-1\)。新增 1 个变量，k增加 1，残差自由度\(n-k-1\)减小。虽然RSS小幅下降，但自由度的惩罚项 \(\frac{RSS}{n-k-1}\) 会变大。当新增变量对降低RSS的贡献很小（纯随机无关特征），自由度惩罚的影响超过 RSS 下降带来的收益，最终 \(\displaystyle\frac{RSS}{n-k-1}\) 变大，导致 \(R^2_{adj}\) 下降。