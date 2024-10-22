import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 定义辅助函数
def softmax(z):
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def tanh_derivative(z):
    return 1 - np.power(np.tanh(z), 2)

# 可视化决策边界
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)# 通过绘制区域来绘制边界
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# 定义预测函数
def predict(X):
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    return softmax(z2)

# 生成数据
X, y = make_moons(n_samples=2000, noise=0.1)

# 可视化数据
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.title('Two Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 二元分类任务，故将 y 进行 one-hot 编码，便于后续的交叉熵损失计算
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1 # 标签 one-hot 编码
# 网络的超参数
n_input = 2      # 输入层大小 (特征数量)
n_hidden = 6     # 隐藏层大小 (可调整)
n_output = 2     # 输出层大小 (类别数量)
learning_rate = 0.001
num_iterations = 100000

# 2层神经网络
# 初始化权重
# np.random.seed(42)
W1 = np.random.randn(n_input, n_hidden)# 
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output)
b2 = np.zeros((1, n_output))

# 训练过程
for i in range(num_iterations):
    # 前向传播
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)  # 激活函数 tanh
    z2 = a1.dot(W2) + b2
    probs = softmax(z2)  # 输出 softmax 概率

    # 计算交叉熵损失
    loss = -np.mean(np.sum(y_onehot * np.log(probs), axis=1))

    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss}")

    # 反向传播
    delta3 = probs - y_onehot
    dW2 = a1.T.dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)

    delta2 = delta3.dot(W2.T) * tanh_derivative(z1)
    dW1 = X.T.dot(delta2)
    db1 = np.sum(delta2, axis=0)

    # 采用梯度下降法更新参数
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# 绘制决策边界
plot_decision_boundary(lambda x: predict(x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()