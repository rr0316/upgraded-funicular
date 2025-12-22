import streamlit as st
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# 页面基础配置
st.set_page_config(
    page_title="中文文本分析工具",
    page_icon="📝",
    layout="wide"
)

# 中文字体适配（兼容云端/本地）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 加载停用词表
def load_stopwords():
    """加载停用词，兼容文件缺失场景"""
    try:
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        st.warning("未检测到停用词文件，使用默认停用词库")
        stopwords = set(["的", "了", "是", "我", "你", "他", "在", "和", "就", "都", "也", "还"])
    return stopwords

# 文本预处理
def preprocess_text(text):
    """文本清洗与分词"""
    stopwords = load_stopwords()
    # 分词
    words = jieba.lcut(text.strip())
    # 过滤停用词、空字符、单字
    valid_words = [word for word in words if word not in stopwords and len(word) >= 2 and word.strip()]
    return valid_words

# 词云生成
def generate_wordcloud(words):
    """生成词云图"""
    word_freq = Counter(words)
    wc = WordCloud(
        width=800,
        height=500,
        background_color="white",
        max_words=100,
        font_path=None,  # 云端自动适配
        colormap="viridis"
    ).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# 主界面逻辑
def main():
    # 侧边栏
    st.sidebar.title("📝 文本分析工具")
    st.sidebar.markdown("### 功能列表")
    st.sidebar.markdown("- 中文分词（Jieba）")
    st.sidebar.markdown("- 词频统计与可视化")
    st.sidebar.markdown("- TF-IDF关键词提取")
    st.sidebar.markdown("- 词云生成")
    st.sidebar.divider()
    st.sidebar.markdown("#### 使用说明")
    st.sidebar.markdown("1. 输入中文文本（支持长文本）")
    st.sidebar.markdown("2. 点击「开始分析」获取结果")
    st.sidebar.markdown("3. 可下载词频数据/词云图")

    # 主标题
    st.title("📊 中文文本分析Web应用")
    st.divider()

    # 文本输入
    text_input = st.text_area(
        label="请输入待分析的中文文本",
        height=200,
        placeholder="示例：人工智能是引领新一轮科技革命和产业变革的重要驱动力，具有溢出带动性很强的“头雁”效应。它不仅推动数字经济发展，还深刻改变着生产生活方式...",
        label_visibility="collapsed"
    )

    # 分析按钮
    if st.button("🚀 开始分析", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("请输入有效文本后再分析！")
            st.stop()
        
        # 1. 文本预处理
        with st.spinner("正在分词处理..."):
            valid_words = preprocess_text(text_input)
            if not valid_words:
                st.error("过滤后无有效词汇，请更换文本（避免全为停用词/单字）")
                st.stop()
        
        # 2. 词频统计
        word_count = Counter(valid_words)
        top20 = word_count.most_common(20)

        # 3. 关键词提取（TF-IDF）
        keywords = jieba.analyse.extract_tags(text_input, topK=10, withWeight=True)

        # 多列展示结果
        col1, col2 = st.columns(2)

        # 列1：词频统计 + 柱状图
        with col1:
            st.subheader("🔤 词频统计（TOP20）")
            # 词频表格
            freq_data = {"词汇": [w[0] for w in top20], "频次": [w[1] for w in top20]}
            st.dataframe(freq_data, hide_index=True, use_container_width=True)
            
            # 词频柱状图
            st.subheader("📈 词频可视化")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar([w[0] for w in top20], [w[1] for w in top20], color="#1f77b4")
            ax.set_xlabel("词汇", fontsize=12)
            ax.set_ylabel("频次", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        # 列2：关键词提取 + 词云
        with col2:
            st.subheader("🏷️ 关键词提取（TF-IDF）")
            keyword_data = {"关键词": [w[0] for w in keywords], "权重": [round(w[1], 4) for w in keywords]}
            st.dataframe(keyword_data, hide_index=True, use_container_width=True)
            
            st.subheader("☁️ 词云生成")
            wc_fig = generate_wordcloud(valid_words)
            st.pyplot(wc_fig)

        # 底部统计信息
        st.divider()
        st.success(f"""
            ✅ 分析完成！
            - 原始文本长度：{len(text_input)} 字符
            - 分词总数：{len(valid_words)} 个
            - 不重复词汇数：{len(word_count)} 个
            - 高频核心词：{top20[0][0]}（频次：{top20[0][1]}）
        """)

if __name__ == "__main__":
    main()
