"""
大众点评用户评分影响因素分析 —— 智能加载版
优先使用 data/sampled_data.csv，若无则从原始数据 data/ratings.csv 开始处理
输出：output/dianping/
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP
import jieba
from collections import Counter
import os, warnings
from tqdm import tqdm
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径设置 ==========
DATA_DIR = 'data'
OUTPUT_DIR = 'output/dianping'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 智能加载采样数据或原始数据 ==========
sample_path = os.path.join(DATA_DIR, 'sampled_data.csv')

if os.path.exists(sample_path):
    print(f"发现已有采样数据 {sample_path}，直接加载...")
    df_sample = pd.read_csv(sample_path)
    print(f"样本记录数: {len(df_sample)}")
else:
    print("未找到采样数据，开始从原始数据加载并处理...")
    # --- 加载原始数据 ---
    rating = pd.read_csv(
        os.path.join(DATA_DIR, 'ratings.csv'),
        nrows=500000,
        usecols=['userId', 'restId', 'rating', 'rating_env', 'rating_flavor',
                 'rating_service', 'timestamp', 'comment']
    )
    print(f"原始记录数: {len(rating)}")

    # --- 清洗 ---
    rating.dropna(subset=['comment'], inplace=True)
    rating.dropna(subset=['rating'], inplace=True)
    rating = rating[(rating['rating'] >= 0) & (rating['rating'] <= 5)]
    rating['timestamp'] = pd.to_datetime(rating['timestamp'], unit='ms', errors='coerce')
    rating.dropna(subset=['timestamp'], inplace=True)
    rating['month'] = rating['timestamp'].dt.to_period('M').dt.to_timestamp()
    rating['weekday'] = rating['timestamp'].dt.weekday
    print(f"清洗后记录数: {len(rating)}")

    if len(rating) == 0:
        raise SystemExit("清洗后无数据，请检查数据文件。")

    # --- 随机抽样 ---
    SAMPLE_SIZE = min(150000, len(rating))
    df_sample = rating.sample(n=SAMPLE_SIZE, random_state=42).copy()

    # --- 就地计算用户/餐厅特征 ---
    print("计算用户活跃度与餐厅热度...")
    df_sample['user_review_count'] = df_sample.groupby('userId')['userId'].transform('count')
    df_sample['user_avg_rating'] = df_sample.groupby('userId')['rating'].transform('mean')
    df_sample['shop_review_count'] = df_sample.groupby('restId')['restId'].transform('count')
    df_sample['shop_avg_rating'] = df_sample.groupby('restId')['rating'].transform('mean')

    # --- 情感分析 ---
    print("正在进行文本情感分析（SnowNLP）...")
    tqdm.pandas(desc="情感计算")
    def get_sentiment(text):
        try:
            return SnowNLP(str(text)).sentiments
        except:
            return 0.5
    df_sample['sentiment'] = df_sample['comment'].progress_apply(get_sentiment)

    # --- 分组构建 ---
    bins = [0, 5, 20, float('inf')]
    df_sample['activity_group'] = pd.cut(df_sample['user_review_count'], bins=bins,
                                         labels=['低 (1-5)', '中 (6-20)', '高 (>20)'], right=True)
    df_sample['shop_heat_group'] = pd.cut(df_sample['shop_review_count'], bins=3,
                                          labels=['低热度', '中热度', '高热度'])

    # --- 保存采样数据供后续直接使用 ---
    df_sample.to_csv(sample_path, index=False)
    print(f"采样数据已保存至 {sample_path}")

# ========== 无论哪种方式，现在 df_sample 已准备好 ==========

# ========== 导出样本数据（可选的额外副本，这里也存一份在 output 方便查阅） ==========
df_sample.to_csv(f'{OUTPUT_DIR}/sampled_data.csv', index=False)

# ========== 图表绘制与数据导出 ==========

# ---- 图1：多维评分随综合评分的变化 ----
print("绘制图1并保存数据...")
sub_fields = ['rating_flavor', 'rating_env', 'rating_service']
sub_data = df_sample.dropna(subset=sub_fields)
mean_sub = sub_data.groupby('rating')[sub_fields].mean().reset_index()
mean_sub.to_csv(f'{OUTPUT_DIR}/multi_dim_scores.csv', index=False)

mean_sub_melted = mean_sub.melt(id_vars='rating', var_name='维度', value_name='平均分')
mean_sub_melted['维度'] = mean_sub_melted['维度'].map({
    'rating_flavor': '口味', 'rating_env': '环境', 'rating_service': '服务'
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=mean_sub_melted, x='rating', y='平均分', hue='维度', palette='Set2', ax=ax)
ax.set_xlabel('综合评分')
ax.set_ylabel('各细分维度平均分')
ax.set_title('图1：不同综合评分下的口味、环境、服务平均得分')
ax.legend(title='维度')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig1_multi_dim_scores.png', dpi=200)
plt.close()

# ---- 图2：情感得分按综合评分的分布 ----
print("绘制图2并保存数据...")
sentiment_stats = df_sample.groupby('rating')['sentiment'].describe()
sentiment_stats.to_csv(f'{OUTPUT_DIR}/sentiment_stats.csv')

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_sample, x='rating', y='sentiment', palette='Blues', ax=ax)
ax.set_xlabel('综合评分')
ax.set_ylabel('文本情感得分 (SnowNLP)')
ax.set_title('图2：不同评分下的评论文本情感得分分布')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig2_sentiment_by_rating.png', dpi=200)
plt.close()

# ---- 图3：用户活跃度与评分 ----
print("绘制图3并保存数据...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

user_agg = df_sample.groupby('userId').agg(
    user_avg_rating=('rating', 'mean'),
    user_review_count=('user_review_count', 'max')
).reset_index()
user_agg['log_reviews'] = np.log1p(user_agg['user_review_count'])
user_plot = user_agg.sample(min(5000, len(user_agg)), random_state=42)
sns.regplot(data=user_plot, x='log_reviews', y='user_avg_rating',
            scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'}, ax=ax1)
ax1.set_xlabel('用户评论数 (对数)')
ax1.set_ylabel('用户平均评分')
ax1.set_title('活跃度与平均评分趋势')

temp = df_sample.dropna(subset=['activity_group'])
mean_act = temp.groupby('activity_group', observed=False)['rating'].mean().reset_index()
sns.barplot(data=mean_act, x='activity_group', y='rating', palette='Set2', ax=ax2)
ax2.set_xlabel('用户活跃度分组')
ax2.set_ylabel('综合评分均值')
ax2.set_title('不同活跃度用户的平均打分')
for i, row in mean_act.iterrows():
    ax2.text(i, row['rating'] + 0.02, f"{row['rating']:.2f}", ha='center')
fig.suptitle('图3：用户活跃度与评分严苛效应', fontsize=14)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig3_user_activity_effect.png', dpi=200, bbox_inches='tight')
plt.close()
mean_act.to_csv(f'{OUTPUT_DIR}/user_activity_stats.csv', index=False)

# ---- 图4：餐厅热度与评分 ----
print("绘制图4并保存数据...")
temp_shop = df_sample.dropna(subset=['shop_heat_group'])
shop_heat_stats = temp_shop.groupby('shop_heat_group')['shop_avg_rating'].describe()
shop_heat_stats.to_csv(f'{OUTPUT_DIR}/shop_heat_stats.csv')

fig, ax = plt.subplots(figsize=(8, 5))
order = ['低热度', '中热度', '高热度']
sns.violinplot(data=temp_shop, x='shop_heat_group', y='shop_avg_rating',
               palette='YlOrRd', order=order, ax=ax, inner='quartile')
ax.set_xlabel('餐厅热度分组')
ax.set_ylabel('餐厅平均评分')
ax.set_title('图4：不同热度餐厅的平均评分分布')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig4_shop_heat_violin.png', dpi=200)
plt.close()

# ---- 图5：词云（含停用词过滤） ----
print("绘制图5并保存数据...")
stopwords = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '他', '她', '它', '们', '那', '什么', '怎么', '为什么', '但', '与',
    '而', '或', '因为', '所以', '如果', '虽然', '可以', '这个', '那个', '还是', '非常',
    '比较', '不过', '而且', '然后', '已经', '就是', '觉得', '知道', '应该', '真的',
    '有点', '一点', '吗', '呢', '吧', '啊', '哦', '嗯', '哈', '么', '没', '还', '被',
    '让', '对', '从', '把', '向', '给', '到', '过'
])

font_path = 'simhei.ttf'
if not os.path.exists(font_path):
    font_path = None

def prepare_word_list(text_series):
    words = []
    for text in text_series:
        seg_list = jieba.cut(str(text))
        words.extend([w for w in seg_list if len(w.strip()) > 1 and w not in stopwords])
    return ' '.join(words)

high_text = df_sample[df_sample['rating'] >= 4]['comment'].astype(str)
low_text = df_sample[df_sample['rating'] <= 2]['comment'].astype(str)

high_words = prepare_word_list(high_text)
low_words = prepare_word_list(low_text)

wc_high = WordCloud(width=800, height=400, background_color='white',
                    font_path=font_path, max_words=100).generate(high_words)
wc_low = WordCloud(width=800, height=400, background_color='white',
                   font_path=font_path, max_words=100).generate(low_words)

# 为了 Matplotlib 中文标题，使用 FontProperties
from matplotlib.font_manager import FontProperties
if font_path and os.path.exists(font_path):
    cn_font = FontProperties(fname=font_path)
else:
    cn_font = None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.imshow(wc_high, interpolation='bilinear')
ax1.axis('off')
title1 = '高分评论 (≥4星)'
title2 = '低分评论 (≤2星)'
if cn_font:
    ax1.set_title(title1, fontproperties=cn_font, fontsize=14)
    ax2.set_title(title2, fontproperties=cn_font, fontsize=14)
else:
    ax1.set_title(title1, fontsize=14)
    ax2.set_title(title2, fontsize=14)

suptitle = '图5：高/低评分评论词云对比'
if cn_font:
    fig.suptitle(suptitle, fontproperties=cn_font, fontsize=16)
else:
    fig.suptitle(suptitle, fontsize=16)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig5_wordcloud.png', dpi=200)
plt.close()

# ---- 图6：方面提及率 ----
print("绘制图6并保存数据...")
aspects = {
    '口味': ['口味', '味道', '好吃', '难吃', '口感', '菜品'],
    '环境': ['环境', '装修', '氛围', '安静', '吵闹', '干净'],
    '服务': ['服务', '服务员', '态度', '热情', '冷漠', '上菜'],
    '价格': ['价格', '性价比', '贵', '便宜', '人均', '实惠']
}

def count_aspects(text_series, aspect_words):
    count = 0
    for text in text_series:
        words = set(jieba.cut(str(text)))
        if any(w in words for w in aspect_words):
            count += 1
    return count / len(text_series)

high_df = df_sample[df_sample['rating'] >= 4]
low_df = df_sample[df_sample['rating'] <= 2]

high_rates = {asp: count_aspects(high_df['comment'], words) for asp, words in aspects.items()}
low_rates = {asp: count_aspects(low_df['comment'], words) for asp, words in aspects.items()}

df_aspect = pd.DataFrame({
    '方面': list(aspects.keys()),
    '高分提及率': list(high_rates.values()),
    '低分提及率': list(low_rates.values())
})
df_aspect.to_csv(f'{OUTPUT_DIR}/aspect_mention_rates.csv', index=False)

df_aspect_melted = df_aspect.melt(id_vars='方面', var_name='评分组', value_name='提及率')
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_aspect_melted, x='方面', y='提及率', hue='评分组',
            palette=['#2ca02c', '#d62728'], ax=ax)
ax.set_title('图6：高分 vs 低分评论中各餐饮方面的提及率')
ax.set_ylabel('提及该方面的评论比例')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig6_aspect_mention_rate.png', dpi=200)
plt.close()

# ---- 图7：回归系数 ----
print("绘制图7并保存数据...")
reg_df = df_sample.dropna(subset=['sentiment', 'user_review_count', 'shop_review_count']).copy()
reg_df['log_user_reviews'] = np.log1p(reg_df['user_review_count'])
reg_df['log_shop_reviews'] = np.log1p(reg_df['shop_review_count'])
features = ['sentiment', 'log_user_reviews', 'log_shop_reviews', 'weekday']
X = reg_df[features]
y = reg_df['rating']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression().fit(X_scaled, y)
coef_df = pd.DataFrame({
    '因素': ['文本情感', '用户活跃度(对数)', '餐厅热度(对数)', '星期几'],
    '标准化系数': model.coef_
}).sort_values('标准化系数', ascending=True)
coef_df.to_csv(f'{OUTPUT_DIR}/regression_coefficients.csv', index=False)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=coef_df, x='标准化系数', y='因素', palette='coolwarm', ax=ax)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('图7：各因素对综合评分的标准化回归系数')
ax.set_xlabel('标准化系数 (绝对值越大影响越大)')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig7_regression_coef.png', dpi=200)
plt.close()

print(f"\n所有图表与数据已保存至 {OUTPUT_DIR}/")
print("分析完成！")