# 线性回归电厂输出预测实验

本项目实现了《线性回归模型实验指导》中的线性回归实验。
 该实验通过训练一个**线性模型（Linear Regression）**，根据四个环境参数预测燃气-蒸汽联合循环电厂的净小时发电量（`PE`）。

模型输入的四个环境参数包括：

- `AT`：环境温度（Ambient Temperature）
- `V`：排气真空度（Exhaust Vacuum）
- `AP`：环境压力（Ambient Pressure）
- `RH`：相对湿度（Relative Humidity）

------

## 项目结构

```
.
├── data/                     # 样例数据集 Folds5x2_pp.csv
│   └── Folds5x2_pp.csv
├── plots/                    # 运行脚本后自动生成的可视化图像
├── requirements.txt          # Python 依赖文件
├── src/
│   └── linear_regression.py  # 命令行实验脚本
└── README.md
```

------

## 环境配置

本实验仅依赖少量常用的科学计算 Python 库。
 建议在虚拟环境中安装依赖：

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 最低支持的 Python 版本为 3.10（根据原实验要求，Python 3.7 及以上版本均可正常运行）。

------

## 运行实验

直接运行主脚本即可完成模型训练、测试集评估及可视化结果生成：

```
python -m src.linear_regression
```

默认设置如下：

- 数据文件：`data/Folds5x2_pp.csv`
- 测试集比例：25%
- 自动打印模型系数与性能指标
- 结果图像保存到：`plots/actual_vs_predicted.png`

查看可用参数：

```
python -m src.linear_regression --help
```

重要参数说明：

- `--data`：指定数据集路径
- `--test-size`：设置测试集占比
- `--random-state`：设置随机种子以保证结果可复现
- `--save-plot`：指定输出图像保存路径（自动创建文件夹）
- `--show-plot`：运行后自动弹出可视化窗口

------

## 样例输出

```
模型系数：
    AT: -0.4235
     V: -0.9308
    AP:  1.1172
    RH: -0.6056
截距:  475.7286

测试集评估指标：
  均方误差 (MSE): 48.7743
  均方根误差 (RMSE): 6.9853
  平均绝对误差 (MAE): 5.8362
  决定系数 (R^2): 0.4429

散点图已保存至: /path/to/repo/plots/actual_vs_predicted.png
```

> 当你修改随机种子、测试集划分比例或使用完整数据集时，以上结果会有所不同。

------

## 可视化说明

生成的散点图与实验指导书中的图形一致：

- 横轴：**实际功率输出值（Actual PE）**
- 纵轴：**模型预测功率输出值（Predicted PE）**
- 红色对角线：理想预测线 `y = x`

通过比较散点与理想线的贴合程度，可以直观评估模型拟合效果。