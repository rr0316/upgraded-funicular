import streamlit as st
import jieba
import jieba.analyse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import warnings
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime

# 全局配置
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="智能文本分析工具（URL/文本双模式）",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------- 核心工具函数 --------------------------
# 中文字体全局适配（兼容云端/本地）
def setup_font():
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial Unicode MS", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams['figure.facecolor'] = 'none'  # 透明背景适配

# 加载停用词表（增强版）
def load_stopwords():
    """加载扩展停用词库，兼容本地/云端"""
    stopwords_basic = {
        "的", "了", "是", "我", "你", "他", "在", "和", "就", "都", "也", "还", "这", "那",
        "把", "被", "为", "于", "之", "而", "则", "虽", "但", "如果", "那么", "因为", "所以",
        "可以", "能够", "应该", "必须", "一个", "一些", "所有", "每个", "没有", "有", "能", "会"
    }
    try:
        # 尝试加载自定义停用词文件
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords_custom = set([line.strip() for line in f if line.strip()])
        stopwords = stopwords_basic.union(stopwords_custom)
    except FileNotFoundError:
        st.warning("未检测到自定义停用词文件，使用内置基础停用词库")
        stopwords = stopwords_basic
    return stopwords

# URL文本爬取（反爬适配）
def crawl_url_text(url):
    """爬取URL页面的正文文本，处理常见反爬和编码问题"""
    try:
        # 模拟浏览器请求头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.baidu.com"
        }
        
        # 发送请求（超时10秒）
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.encoding = response.apparent_encoding  # 自动识别编码
        
        # 解析正文（适配常见网页结构）
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除广告、导航、脚本等无关标签
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "ad"]):
            tag.decompose()
        
        # 优先提取常见正文标签内容
        content_tags = ["article", "div[class*=content]", "div[class*=article]", 
                        "div[id*=content]", "main", "p"]
        text_parts = []
        
        for tag in content_tags:
            elements = soup.select(tag)
            if elements:
                for elem in elements:
                    text = elem.get_text(strip=True, separator="\n")
                    if len(text) > 50:  # 过滤短文本（避免导航/广告）
                        text_parts.append(text)
        
        # 兜底：提取所有文本并过滤
        if not text_parts:
            all_text = soup.get_text(strip=True, separator="\n")
            # 按行过滤，保留长行文本
            text_parts = [line for line in all_text.split("\n") if len(line) > 30]
        
        # 合并并清洗文本
        raw_text = "\n".join(text_parts)
        # 移除多余空格、换行、特殊符号
        clean_text = re.sub(r"\s+", " ", raw_text)
        clean_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·]", "", clean_text)
        
        if len(clean_text) < 100:
            st.error(f"URL爬取失败：页面有效文本过短（仅{len(clean_text)}字符）")
            return None
        return clean_text
    
    except requests.exceptions.Timeout:
        st.error("URL请求超时：请检查网址是否可访问，或目标网站有反爬限制")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"URL访问失败：{str(e)}")
        return None
    except Exception as e:
        st.error(f"文本解析失败：{str(e)}")
        return None

# 文本预处理（支持分词粒度切换）
def preprocess_text(text, seg_mode="精确模式"):
    """
    文本清洗与分词
    :param text: 原始文本
    :param seg_mode: 分词模式：精确模式/全模式/搜索引擎模式
    :return: 过滤后的有效词汇列表
    """
    stopwords = load_stopwords()
    
    # 设置分词模式
    if seg_mode == "全模式":
        words = jieba.lcut(text.strip(), cut_all=True)
    elif seg_mode == "搜索引擎模式":
        words = jieba.lcut_for_search(text.strip())
    else:  # 精确模式（默认）
        words = jieba.lcut(text.strip())
    
    # 多层过滤：停用词、空字符、单字、数字、英文（可配置）
    valid_words = []
    for word in words:
        word_strip = word.strip()
        # 过滤条件
        if (word_strip not in stopwords and 
            len(word_strip) >= 2 and 
            not word_strip.isdigit() and 
            not all(ord(c) < 128 for c in word_strip)):  # 过滤纯英文/数字
            valid_words.append(word_strip)
    
    return valid_words

# 高级词云生成（多样式可选）
def generate_wordcloud(words, style="渐变蓝紫"):
    """
    生成定制化词云图
    :param words: 词汇列表
    :param style: 样式：渐变蓝紫/活力橙红/森系绿/复古黄
    :return: 词云图对象
    """
    word_freq = Counter(words)
    # 配色方案映射
    colormap_map = {
        "渐变蓝紫": "viridis",
        "活力橙红": "plasma",
        "森系绿": "Greens",
        "复古黄": "YlOrBr"
    }
    # 词云配置
    wc = WordCloud(
        width=900,
        height=600,
        background_color="white",
        max_words=150,
        font_path=None,  # 云端自动适配
        colormap=colormap_map.get(style, "viridis"),
        contour_width=1,
        contour_color="lightgray",
        random_state=42  # 固定随机种子，保证结果一致
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig

# 高频词趋势分析（滑动窗口）
def generate_trend_chart(words, window_size=50):
    """
    生成高频词滑动窗口趋势图
    :param words: 词汇列表
    :param window_size: 滑动窗口大小
    :return: 趋势图对象
    """
    # 获取TOP5高频词
    top5_words = [w[0] for w in Counter(words).most_common(5)]
    if len(top5_words) < 1:
        return None
    
    # 滑动窗口统计词频
    windows = [words[i:i+window_size] for i in range(0, len(words)-window_size+1, window_size//2)]
    window_pos = [i * (window_size//2) + window_size//2 for i in range(len(windows))]
    trend_data = {word: [] for word in top5_words}
    
    for window in windows:
        window_count = Counter(window)
        for word in top5_words:
            trend_data[word].append(window_count.get(word, 0))
    
    # 绘制趋势图
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, word in enumerate(top5_words):
        ax.plot(window_pos, trend_data[word], label=word, color=colors[i], linewidth=2, marker="o", markersize=4)
    
    ax.set_xlabel("文本位置（词汇数）", fontsize=12)
    ax.set_ylabel("词频（滑动窗口）", fontsize=12)
    ax.set_title("TOP5高频词趋势分析（滑动窗口：{}词）".format(window_size), fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# -------------------------- 主界面逻辑 --------------------------
def main():
    setup_font()  # 初始化字体
    
    # 侧边栏配置区
    with st.sidebar:
        st.title("⚙️ 分析配置")
        
        # 输入模式切换
        input_mode = st.radio("输入模式", ["URL模式", "纯文本模式"], index=0)
        
        # 分词模式选择
        seg_mode = st.selectbox(
            "分词模式",
            ["精确模式", "全模式", "搜索引擎模式"],
            help="精确模式：最常用，分词准确；全模式：穷尽所有可能；搜索引擎模式：适合搜索引擎分词"
        )
        
        # 词云样式选择
        wc_style = st.selectbox("词云样式", ["渐变蓝紫", "活力橙红", "森系绿", "复古黄"])
        
        # 高级选项折叠框
        with st.expander("🔧 高级选项", expanded=False):
            min_word_len = st.slider("最小词汇长度", 2, 4, 2, help="过滤小于该长度的词汇")
            top_n = st.slider("展示TOP词频数量", 10, 50, 20)
            window_size = st.slider("滑动窗口大小（趋势图）", 30, 100, 50)
        
        st.divider()
        st.markdown("### 📖 使用指南")
        st.markdown("""
        1. URL模式：输入文章链接（支持新闻、博客、公众号等）
        2. 纯文本模式：直接粘贴中文文本
        3. 配置分词/可视化参数，点击「开始分析」
        4. 支持导出词频数据、下载可视化图表
        """)
        st.markdown("### 🚨 注意事项")
        st.markdown("""
        - URL模式仅支持公开可访问的网页
        - 部分网站有反爬机制，可能爬取失败
        - 建议文本长度≥200字符，分析结果更准确
        """)

    # 主界面标题
    st.title("📊 智能文本分析系统（URL增强版）")
    st.markdown("#### 支持URL爬取+多维度文本分析+可视化导出")
    st.divider()

    # 输入区域
    col_input, col_help = st.columns([4, 1])
    with col_input:
        if input_mode == "URL模式":
            url_input = st.text_input(
                label="📎 输入文章URL",
                placeholder="示例：https://www.xxx.com/article/123.html",
                label_visibility="collapsed"
            )
            # URL验证
            if url_input and not urlparse(url_input).scheme:
                st.warning("URL格式错误：请以http/https开头")
                url_input = ""
        else:
            text_input = st.text_area(
                label="📝 输入待分析的中文文本",
                height=250,
                placeholder="示例：人工智能是引领新一轮科技革命和产业变革的重要驱动力...",
                label_visibility="collapsed"
            )
    
    # 分析按钮
    analyze_btn = st.button("🚀 开始深度分析", type="primary", use_container_width=True)
    
    # 分析逻辑
    if analyze_btn:
        # 1. 获取文本（URL模式/纯文本模式）
        if input_mode == "URL模式":
            if not url_input:
                st.warning("请输入有效的文章URL！")
                st.stop()
            with st.spinner("🔍 正在爬取URL文本..."):
                raw_text = crawl_url_text(url_input)
                if not raw_text:
                    st.stop()
        else:
            if not text_input.strip():
                st.warning("请输入有效文本！")
                st.stop()
            raw_text = text_input.strip()
        
        st.success(f"✅ 文本加载完成（原始长度：{len(raw_text)} 字符）")
        
        # 2. 文本预处理
        with st.spinner("⚡ 正在分词+清洗文本..."):
            valid_words = preprocess_text(raw_text, seg_mode)
            if not valid_words:
                st.error("过滤后无有效词汇，请调整停用词或文本内容！")
                st.stop()
        
        # 3. 核心分析计算
        word_count = Counter(valid_words)
        total_words = len(valid_words)
        unique_words = len(word_count)
        top_words = word_count.most_common(top_n)
        
        # TF-IDF关键词提取（增强版）
        jieba.analyse.set_stop_words("stopwords.txt") if "stopwords.txt" in st.session_state else None
        keywords = jieba.analyse.extract_tags(
            raw_text, 
            topK=10, 
            withWeight=True,
            allowPOS=("n", "v", "adj")  # 仅提取名词、动词、形容词
        )
        
        # 4. 多维度结果展示
        st.divider()
        st.subheader("📈 文本基础统计")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("原始文本长度", f"{len(raw_text)} 字符")
        with stat_col2:
            st.metric("有效分词数", f"{total_words} 个")
        with stat_col3:
            st.metric("不重复词汇数", f"{unique_words} 个")
        with stat_col4:
            st.metric("词汇丰富度", f"{round(unique_words/total_words*100, 2)} %")
        
        # 5. 高频词分析（多标签页展示）
        tab1, tab2, tab3, tab4 = st.tabs(["📊 词频统计", "🏷️ 关键词提取", "☁️ 词云可视化", "📉 趋势分析"])
        
        # 标签页1：词频统计 + 柱状图 + 导出
        with tab1:
            st.subheader(f"TOP{top_n} 高频词统计")
            # 生成词频DataFrame
            freq_df = pd.DataFrame(top_words, columns=["词汇", "频次"])
            freq_df["占比(%)"] = round(freq_df["频次"] / total_words * 100, 2)
            
            # 展示表格
            st.dataframe(freq_df, hide_index=True, use_container_width=True)
            
            # 导出按钮
            csv_data = freq_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="💾 导出词频数据（CSV）",
                data=csv_data,
                file_name=f"词频统计_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # 高频词柱状图（横向，更易读）
            st.subheader("📊 高频词可视化")
            fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(top_words[:15]))  # 展示TOP15
            ax_bar.barh(y_pos, [w[1] for w in top_words[:15]], color="#1f77b4")
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels([w[0] for w in top_words[:15]])
            ax_bar.set_xlabel("频次", fontsize=12)
            ax_bar.set_title(f"TOP15 高频词柱状图", fontsize=14)
            ax_bar.grid(axis="x", alpha=0.3)
            st.pyplot(fig_bar)
        
        # 标签页2：关键词提取（TF-IDF）
        with tab2:
            st.subheader("TF-IDF 关键词提取（TOP10）")
            keyword_df = pd.DataFrame(keywords, columns=["关键词", "权重"])
            keyword_df["权重"] = round(keyword_df["权重"], 4)
            st.dataframe(keyword_df, hide_index=True, use_container_width=True)
            
            # 关键词权重饼图
            fig_pie, ax_pie = plt.subplots(figsize=(10, 10))
            top8_keywords = keyword_df.head(8)
            ax_pie.pie(
                top8_keywords["权重"],
                labels=top8_keywords["关键词"],
                autopct="%1.2f%%",
                startangle=90,
                colors=plt.cm.viridis(np.linspace(0, 1, len(top8_keywords)))
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)
        
        # 标签页3：词云可视化（可下载）
        with tab3:
            st.subheader("🎨 词云可视化（{}）".format(wc_style))
            wc_fig = generate_wordcloud(valid_words, wc_style)
            st.pyplot(wc_fig)
            
            # 保存词云图并提供下载
            import io
            buf = io.BytesIO()
            wc_fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="💾 下载词云图（PNG）",
                data=buf,
                file_name=f"词云_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        # 标签页4：趋势分析（滑动窗口）
        with tab4:
            st.subheader("📉 高频词趋势分析")
            trend_fig = generate_trend_chart(valid_words, window_size)
            if trend_fig:
                st.pyplot(trend_fig)
                
                # 趋势图下载
                trend_buf = io.BytesIO()
                trend_fig.savefig(trend_buf, format="png", dpi=300, bbox_inches="tight")
                trend_buf.seek(0)
                st.download_button(
                    label="💾 下载趋势图（PNG）",
                    data=trend_buf,
                    file_name=f"趋势分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("文本长度不足，无法生成趋势图")
        
        # 6. 分析报告总结
        st.divider()
        st.subheader("📋 分析报告总结")
        summary_text = f"""
        **分析时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **输入模式**：{input_mode}
        **分词模式**：{seg_mode}
        **核心结论**：
        1. 文本核心词汇：{top_words[0][0]}（频次：{top_words[0][1]}，占比：{round(top_words[0][1]/total_words*100, 2)}%）
        2. 词汇丰富度：{round(unique_words/total_words*100, 2)}%（数值越高，文本词汇多样性越好）
        3. 核心关键词（TF-IDF）：{', '.join([w[0] for w in keywords[:5]])}
        4. 文本长度：{len(raw_text)}字符，有效分词：{total_words}个，文本信息量充足
        """
        st.markdown(summary_text)
        
        # 报告导出
        report_content = f"""
# 智能文本分析报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
输入模式：{input_mode}
分词模式：{seg_mode}

## 基础统计
- 原始文本长度：{len(raw_text)} 字符
- 有效分词数：{total_words} 个
- 不重复词汇数：{unique_words} 个
- 词汇丰富度：{round(unique_words/total_words*100, 2)} %

## TOP10 高频词
{freq_df.head(10).to_string(index=False)}

## TOP10 关键词（TF-IDF）
{keyword_df.to_string(index=False)}

## 核心结论
1. 文本核心词汇：{top_words[0][0]}（频次：{top_words[0][1]}，占比：{round(top_words[0][1]/total_words*100, 2)}%）
2. 核心关键词：{', '.join([w[0] for w in keywords[:5]])}
3. 文本词汇丰富度{round(unique_words/total_words*100, 2)}%，文本信息维度：{"丰富" if unique_words/total_words > 0.2 else "一般" if unique_words/total_words > 0.1 else "单一"}
        """
        st.download_button(
            label="📄 导出分析报告（TXT）",
            data=report_content,
            file_name=f"文本分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
