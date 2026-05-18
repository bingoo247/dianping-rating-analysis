# 大众点评用户评分影响因素分析

基于公开数据集 yf_dianping 的餐饮评分影响因素研究，涵盖数据清洗、特征构建、情感分析、可视化与回归建模。  
本项目为《数据挖掘与可视化》课程论文的配套代码仓库。

## 主要发现
- 口味评分是综合评分的最大贡献维度，服务评分是低分评论的主要“短板”。
- 文本情感与用户评分高度正相关，但低分评论存在“低分高情”的离群值。
- 高活跃用户存在“严苛效应”（平均评分更低）。
- 高热度餐厅不仅评分较高，评价一致性也更强。
- 服务问题是引发差评的最常见原因。

## 仓库结构
```
.
├── data/                        # 数据目录
│   ├── sampled_data.csv          # 已清洗的采样数据（15万条），可直接用于绘图
│   ├── restaurants.csv           # 餐馆信息
│   ├── links.csv                 # 餐馆ID映射
│   └── ratings.csv               # 原始评分数据（需自行下载，约1.75GB）
├── output/dianping/              # 输出目录（图表与统计CSV）
│   ├── fig1~fig7.png             # 7张高清图表
│   └── *.csv                     # 各图表对应的统计数据
├── dianping.py                   # 主分析脚本（智能加载采样或原始数据）
├── requirements.txt              # Python依赖
├── .gitignore
└── README.md
```

## 数据获取
本项目使用 **[yf_dianping](https://hyper.ai/datasets/29993)** 公开数据集。  
仓库已包含清洗后的 **sampled_data.csv**，可直接运行生成全部图表。  
如需从原始数据开始完整处理，请手动下载 `ratings.csv`、`restaurants.csv`、`links.csv` 并放入 `data/` 目录。

## 环境配置
1. 克隆仓库
   ```bash
   git clone https://github.com/你的用户名/仓库名.git
   cd 仓库名
   ```
2. 创建虚拟环境（推荐）
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
### 直接生成图表（使用采样数据）
确保 `data/sampled_data.csv` 存在，然后运行：
```bash
python dianping.py
```
脚本会自动检测采样文件，跳过数据清洗与情感分析，直接绘图并导出统计CSV。

### 从原始数据重新处理
1. 将下载的 `ratings.csv`、`restaurants.csv`、`links.csv` 放入 `data/` 目录。
2. 运行 `python dianping.py`，脚本将执行：
   - 读取前50万条记录
   - 清洗并随机抽样15万条
   - 计算用户/餐厅特征
   - SnowNLP 情感分析（带进度条）
   - 生成7张图表和6份统计文件
3. 首次运行后会在 `data/` 下生成 `sampled_data.csv`，后续可直接使用。

## 主要依赖
- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- snownlp (情感分析)
- jieba (中文分词)
- wordcloud (词云)
- tqdm (进度条)
- scikit-learn (回归模型)

详见 `requirements.txt`。

## 许可
本项目仅用于学术研究，数据集版权归原作者及平台所有。代码部分采用 [MIT License](LICENSE)。

## 引用
如果使用了本仓库的代码或结果，请注明：
```
@misc{dianping-rating-analysis,
  author = {你的姓名},
  title = {大众点评用户评分影响因素分析},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/你的用户名/仓库名}}
}
```
