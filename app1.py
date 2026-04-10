












import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import google.generativeai as genai
import plotly.express as px
import os
from textblob import TextBlob

# ==========================================
# 1. 页面级配置 & 全局 CSS 注入 
# ==========================================
st.set_page_config(page_title="Bubble Shooter 智能体检报告", page_icon="📱", layout="wide")

st.markdown("""
<style>
    .report-header { font-size: 2.5em; font-weight: 800; border-bottom: 3px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;}
    .section-title { font-size: 1.5em; font-weight: 700; color: #1f77b4; margin-top: 30px; margin-bottom: 15px;}
    div[data-testid="metric-container"] { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1f77b4;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心功能函数
# ==========================================
def init_gemini(api_key):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash') 
    except Exception as e:
        st.error(f"Gemini 初始化失败: {e}")
        return None

def analyze_text_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['content'] = df['content'].fillna('')
        df['at'] = pd.to_datetime(df['at'])
        df['date'] = df['at'].dt.date
        df['week'] = df['at'].dt.isocalendar().week
        
        df['star_rating'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
        df['nlp_sentiment_score'] = df['content'].apply(analyze_text_sentiment)
        df['true_sentiment'] = pd.cut(df['nlp_sentiment_score'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
        
        return df
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None

def get_zeus_style_insight(model, df):
    sample_reviews = df[df['content'].str.len() > 20].sample(min(100, len(df)))['content'].tolist()
    text_to_analyze = "\n".join(sample_reviews)
    
    prompt = f"""
    作为传音高级游戏数据分析师，请阅读以下抽样的真实玩家评论数据，生成一份结构化商业诊断报告。
    你必须严格使用以下 Markdown 结构和表情符号进行输出，保持专业、客观，避免空话：

    ### 🤖 AI 深度洞察
    **📋 执行摘要**
    (用一段话概括当前版本的核心口碑盘面与最致命危机)

    **💡 关键发现**
    (列举3-4条最突出的痛点或爽点，必须带有极强的游戏业务感，如“商业化变现”、“数值平衡”、“系统崩溃”等，并引用玩家原话作为佐证)

    **👥 核心用户画像**
    (描述最容易因为上述问题退坑的玩家特征)

    ### 🚀 战略改进建议
    * 🔧 **短期（1-3个月，针对研发与运营）**：(列出亟待修复的 Bug 或必须调整的策略)
    * 🎯 **中期（3-6个月，针对策划与发行）**：(系统玩法或商业化节奏的优化建议)
    * 🌟 **长期（6-12个月，针对大盘生态）**：(游戏核心机制或长期留存的升级方向)

    评论数据源:
    {text_to_analyze}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 分析时出错: {e}"

# ==========================================
# 3. 网页主程序 (自动加载本地数据)
# ==========================================
YOUR_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# 【核心修改区】：将之前侧边栏的文件上传，改为直接指定本地文件名
DATA_FILENAME = "linkdesks.pop.bubblegames.bubbleshooter_all_reviews.csv"

# 检查文件是否存在
if not os.path.exists(DATA_FILENAME):
    st.error(f"🚨 找不到数据文件：`{DATA_FILENAME}`。请确保它已经与本代码文件存放在同一个文件夹中！")
else:
    # 自动加载数据并渲染网页
    with st.spinner("正在加载预置业务数据..."):
        df = load_and_clean_data(DATA_FILENAME)
        
    if df is not None:
        st.markdown('<div class="report-header">📱 Bubble Shooter VOC 智能体检报告</div>', unsafe_allow_html=True)
        st.caption(f"分析样本区间: {df['date'].min()} 至 {df['date'].max()} | 数据源: Google Play")
        
        # --- 模块 1: 📊 情感分析 ---
        st.markdown('<div class="section-title">📊 情感分析与基础指标</div>', unsafe_allow_html=True)
        
        avg_score = df['score'].mean()
        avg_sentiment = df['nlp_sentiment_score'].mean()
        pos_pct = (len(df[df['true_sentiment'] == 'Positive']) / len(df)) * 100
        neg_pct = (len(df[df['true_sentiment'] == 'Negative']) / len(df)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("分析评价数", f"{len(df):,}")
        col2.metric("平均星级评分", f"{avg_score:.2f} / 5.0")
        col3.metric("NLP 情感均分", f"{avg_sentiment:.2f}", delta=">0为正向, <0为负向", delta_color="off")
        col4.metric("真实负面口碑占比", f"{neg_pct:.1f}%", delta=f"好评 {pos_pct:.1f}%", delta_color="inverse")

        # --- 模块 2: 📈 趋势分析 ---
        st.markdown('<div class="section-title">📈 周度指标趋势分析</div>', unsafe_allow_html=True)
        trend_col1, trend_col2 = st.columns([2, 1])
        
        with trend_col1:
            weekly_trend = df.groupby('week').agg(
                avg_score=('score', 'mean'),
                review_count=('score', 'count')
            ).reset_index()
            fig_trend = px.line(weekly_trend, x='week', y='avg_score', text=weekly_trend['avg_score'].round(2),
                                title="自然周平均星级走势", markers=True)
            fig_trend.update_traces(textposition="top center")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with trend_col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.dataframe(weekly_trend.rename(columns={'week':'周次', 'avg_score':'平均评分', 'review_count':'评价数量'}), hide_index=True)

# --- 模块 3: 🤖 AI 深度洞察与战略规划 ---
        st.markdown('<div class="section-title">🤖 AI 深度洞察与战略建议</div>', unsafe_allow_html=True)
        
        # 1. 初始化会话缓存 (Session State)
        if "ai_report_cache" not in st.session_state:
            st.session_state.ai_report_cache = None

        # 2. 并排设置操作按钮
        btn_col1, btn_col2 = st.columns([2, 8])
        with btn_col1:
            # 如果缓存为空显示"生成"，如果不为空显示"重新生成"
            btn_label = "⚡ 立即生成报告" if st.session_state.ai_report_cache is None else "🔄 重新生成报告"
            generate_clicked = st.button(btn_label, type="primary")
            
        with btn_col2:
            # 只有在有缓存时，才显示清除缓存的按钮
            if st.session_state.ai_report_cache is not None:
                if st.button("🗑️ 清除当前缓存"):
                    st.session_state.ai_report_cache = None
                    st.rerun() # 强制刷新页面

        # 3. 处理按钮点击逻辑 (仅在此处消耗 Token)
        if generate_clicked:
            if not YOUR_GEMINI_API_KEY:
                st.error("未配置 API Key，无法呼叫 AI 大模型。")
            else:
                with st.spinner('AI 正在交叉验证数据，生成高管级汇报纪要...'):
                    report_content = get_zeus_style_insight(gemini_model, df)
                    
                    # 检查是否是 API 报错，如果不包含报错信息，则写入缓存
                    if "AI 分析时出错" not in report_content:
                        st.session_state.ai_report_cache = report_content
                        st.success("✅ 报告生成完毕，已自动存入本地缓存！")
                    else:
                        st.error(report_content)

        # 4. 独立的数据渲染层 (脱离按钮点击事件，只要缓存有数据就始终显示)
        if st.session_state.ai_report_cache:
            st.markdown(
                f"<div style='background-color: rgba(128, 128, 128, 0.1); padding: 25px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2);'>"
                f"{st.session_state.ai_report_cache}"
                f"</div>", 
                unsafe_allow_html=True
            )
        else:
            st.info("👆 点击上方按钮，消耗 API Token 生成诊断报告。生成后将自动缓存，不受页面筛选刷新影响。")

        
        # # --- 模块 3: 🤖 AI 深度洞察与战略规划 ---
        # st.markdown('<div class="section-title">🤖 AI 深度洞察与战略建议</div>', unsafe_allow_html=True)
        
        # # ⚠️ 注意：这里依然保留了按钮触发，避免每次刷新网页都耗费 API Token
        # if st.button("⚡ 立即生成专家级体检报告", type="primary"):
        #     if not YOUR_GEMINI_API_KEY:
        #         st.error("未配置 API Key，无法呼叫 AI 大模型。")
        #     else:
        #         with st.spinner('AI 正在交叉验证数据，生成高管级汇报纪要...'):
        #             report_content = get_zeus_style_insight(gemini_model, df)
        #             st.success("报告生成完毕")
        #             # 已修复深色模式看不见文字的问题
        #             st.markdown(f"<div style='background-color: rgba(128, 128, 128, 0.1); padding: 25px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2);'>{report_content}</div>", unsafe_allow_html=True)
# --- 模块 4: 词云辅助 (已取消折叠) ---
        st.markdown('<div class="section-title">👁️ 原始文本语义聚类 (词云)</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("负面高频词")
            neg_words = " ".join(df[df['true_sentiment'] == 'Negative']['content'].tolist())
            if neg_words:
                wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
                fig_n, ax_n = plt.subplots(figsize=(8, 4))
                ax_n.imshow(wordcloud_neg)
                ax_n.axis('off')
                st.pyplot(fig_n)
            else:
                st.info("暂无足够的负面评价生成词云")

        with c2:
            st.subheader("正面高频词")
            pos_words = " ".join(df[df['true_sentiment'] == 'Positive']['content'].tolist())
            if pos_words:
                wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
                fig_p, ax_p = plt.subplots(figsize=(8, 4))
                ax_p.imshow(wordcloud_pos)
                ax_p.axis('off')
                st.pyplot(fig_p)
            else:
                st.info("暂无足够的正面评价生成词云")

        # --- 模块 5: 原始数据下钻与动态筛选 (已取消折叠) ---
        st.markdown('<div class="section-title">🔍 原始文本数据下钻</div>', unsafe_allow_html=True)
        st.info("支持组合筛选与下载，按时间倒序排列")

        # 筛选器布局
        filter_c1, filter_c2, filter_c3 = st.columns([1, 1, 1.5])
        
        with filter_c1:
            score_filter = st.multiselect(
                "⭐ 筛选星级 (Score):", 
                options=[5, 4, 3, 2, 1], 
                default=[1, 2] 
            )
            
        with filter_c2:
            available_versions = df['reviewCreatedVersion'].dropna().unique().tolist()
            version_filter = st.multiselect(
                "📱 筛选应用版本 (Version):", 
                options=available_versions,
                default=[] 
            )
            
        with filter_c3:
            search_kw = st.text_input("🔑 关键词检索 (例如输入 '闪退' 或 '黑屏'):", "")

        # 动态过滤数据
        filtered_df = df[df['score'].isin(score_filter)]
        if version_filter:
            filtered_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(version_filter)]
        if search_kw:
            filtered_df = filtered_df[filtered_df['content'].str.contains(search_kw, case=False, na=False)]
        if 'at' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by='at', ascending=False)

        st.caption(f"当前筛选条件下，共找到 **{len(filtered_df)}** 条玩家反馈。")
        
        # 数据表渲染
        st.dataframe(
            filtered_df[['at', 'score', 'content', 'reviewCreatedVersion']], 
            use_container_width=True,
            height=600, # 既然不折叠了，可以适当增加高度方便查阅
            hide_index=True, 
            column_config={
                "at": st.column_config.DatetimeColumn("评论时间", format="YYYY-MM-DD HH:mm"),
                "score": st.column_config.NumberColumn("评分", format="%d ⭐"),
                "content": st.column_config.TextColumn("玩家原始评论", width="large"),
                "reviewCreatedVersion": st.column_config.TextColumn("发生版本")
            }
        )

#         # --- 模块 4: 词云辅助 ---
#         with st.expander("👁️ 查看原始文本语义聚类 (词云)"):
#             c1, c2 = st.columns(2)
#             with c1:
#                 neg_words = " ".join(df[df['true_sentiment'] == 'Negative']['content'].tolist())
#                 if neg_words:
#                     wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
#                     fig_n, ax_n = plt.subplots(figsize=(8, 4))
#                     ax_n.imshow(wordcloud_neg)
#                     ax_n.axis('off')
#                     st.pyplot(fig_n)
#             with c2:
#                 pos_words = " ".join(df[df['true_sentiment'] == 'Positive']['content'].tolist())
#                 if pos_words:
#                     wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
#                     fig_p, ax_p = plt.subplots(figsize=(8, 4))
#                     ax_p.imshow(wordcloud_pos)
#                     ax_p.axis('off')
#                     st.pyplot(fig_p)

# # --- 模块 5: 原始数据下钻与动态筛选 ---
#         st.markdown('<div class="section-title">🔍 原始文本数据下钻</div>', unsafe_allow_html=True)
        
#         # 默认展开，方便老板和运营直接看到
#         with st.expander("点击展开查看具体玩家评论 (支持组合筛选与下载)", expanded=True):
            
#             # 使用三栏设计，增加关键词检索，提升实用性
#             filter_c1, filter_c2, filter_c3 = st.columns([1, 1, 1.5])
            
#             with filter_c1:
#                 # 星级筛选器
#                 score_filter = st.multiselect(
#                     "⭐ 筛选星级 (Score):", 
#                     options=[5, 4, 3, 2, 1], 
#                     default=[1, 2] 
#                 )
                
#             with filter_c2:
#                 # 版本号筛选器
#                 available_versions = df['reviewCreatedVersion'].dropna().unique().tolist()
#                 version_filter = st.multiselect(
#                     "📱 筛选应用版本 (Version):", 
#                     options=available_versions,
#                     default=[] 
#                 )
                
#             with filter_c3:
#                 # 新增：关键词检索（支持多词或正则，找 bug 神器）
#                 search_kw = st.text_input("🔑 关键词检索 (例如输入 '闪退' 或 '黑屏'):", "")

#             # 核心动作：根据用户的选择，动态过滤数据
#             filtered_df = df[df['score'].isin(score_filter)]
            
#             if version_filter:
#                 filtered_df = filtered_df[filtered_df['reviewCreatedVersion'].isin(version_filter)]
                
#             if search_kw:
#                 # 使用 case=False 忽略大小写，na=False 处理空值
#                 filtered_df = filtered_df[filtered_df['content'].str.contains(search_kw, case=False, na=False)]
            
#             # 新增：按时间倒序排列，保证看到的是最新反馈
#             if 'at' in filtered_df.columns:
#                 filtered_df = filtered_df.sort_values(by='at', ascending=False)

#             # 顶部增加数据量提示（动态反馈）
#             st.caption(f"当前筛选条件下，共找到 **{len(filtered_df)}** 条玩家反馈。")
            
#             # 展示最终的数据表 (补全 height，并引入 column_config 让表格更炫酷)
#             st.dataframe(
#                 filtered_df[['at', 'score', 'content', 'reviewCreatedVersion']], 
#                 use_container_width=True,
#                 height=400, # 补全你的代码：固定高度，超出内部滚动
#                 hide_index=True, # 隐藏默认的数字索引，让表格更干净
#                 column_config={
#                     "at": st.column_config.DatetimeColumn(
#                         "评论时间",
#                         format="YYYY-MM-DD HH:mm", # 格式化时间，去掉秒等冗余信息
#                     ),
#                     "score": st.column_config.NumberColumn(
#                         "评分",
#                         help="玩家给出的星级评分",
#                         format="%d ⭐", # 炫酷点：将数字直接渲染为带有星星符号的文本
#                     ),
#                     "content": st.column_config.TextColumn(
#                         "玩家原始评论",
#                         width="large", # 给文本列分配最大宽度
#                     ),
#                     "reviewCreatedVersion": st.column_config.TextColumn(
#                         "发生版本"
#                     )
#                 }
#             )





# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import re
# import google.generativeai as genai
# import plotly.express as px
# import os
# from textblob import TextBlob  # 新增：用于真实文本情感计算

# # ==========================================
# # 1. 页面级配置 & 全局 CSS 注入 (强化企业级报告感)
# # ==========================================
# st.set_page_config(page_title="游戏 VOC 智能诊断报告", page_icon="📱", layout="wide")

# st.markdown("""
# <style>
#     .report-header { font-size: 2.5em; font-weight: 800; border-bottom: 3px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;}
#     .section-title { font-size: 1.5em; font-weight: 700; color: #1f77b4; margin-top: 30px; margin-bottom: 15px;}
#     div[data-testid="metric-container"] { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 4px solid #1f77b4;}
# </style>
# """, unsafe_allow_html=True)

# # ==========================================
# # 2. 核心功能函数 (引入 TextBlob 情感计算)
# # ==========================================
# def init_gemini(api_key):
#     if not api_key: return None
#     try:
#         genai.configure(api_key=api_key)
#         # 兼容性处理：如果 flash 不可用，自动降级到 pro
#         return genai.GenerativeModel('gemini-2.5-flash') 
#     except Exception as e:
#         st.error(f"Gemini 初始化失败: {e}")
#         return None

# def analyze_text_sentiment(text):
#     """利用 TextBlob 计算真实文本情感 (-1.0 到 1.0)"""
#     if not isinstance(text, str) or not text.strip():
#         return 0.0
#     return TextBlob(text).sentiment.polarity

# @st.cache_data
# def load_and_clean_data(file_obj):
#     try:
#         df = pd.read_csv(file_obj)
#         df['content'] = df['content'].fillna('')
#         df['at'] = pd.to_datetime(df['at'])
#         df['date'] = df['at'].dt.date
#         df['week'] = df['at'].dt.isocalendar().week
        
#         # 【核心升级】：双维情感判定
#         # 1. 业务星级评判
#         df['star_rating'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
#         # 2. 文本真实情感计算
#         df['nlp_sentiment_score'] = df['content'].apply(analyze_text_sentiment)
#         # 根据极性重新打标
#         df['true_sentiment'] = pd.cut(df['nlp_sentiment_score'], bins=[-1.1, -0.1, 0.1, 1.1], labels=['Negative', 'Neutral', 'Positive'])
        
#         return df
#     except Exception as e:
#         st.error(f"数据解析失败: {e}")
#         return None

# def get_zeus_style_insight(model, df):
#     """全面模仿 Zeus 报告的 Prompt 架构"""
#     # 抽取具有代表性的评论（情感极端的长评论）
#     sample_reviews = df[df['content'].str.len() > 20].sample(min(100, len(df)))['content'].tolist()
#     text_to_analyze = "\n".join(sample_reviews)
    
#     prompt = f"""
#     作为传音高级游戏数据分析师，请阅读以下抽样的真实玩家评论数据，生成一份结构化商业诊断报告。
#     你必须严格使用以下 Markdown 结构和表情符号进行输出，保持专业、客观，避免空话：

#     ### 🤖 AI 深度洞察
#     **📋 执行摘要**
#     (用一段话概括当前版本的核心口碑盘面与最致命危机)

#     **💡 关键发现**
#     (列举3-4条最突出的痛点或爽点，必须带有极强的游戏业务感，如“商业化变现”、“数值平衡”、“系统崩溃”、“操作反馈”等，并引用玩家原话作为佐证)

#     **👥 核心用户画像**
#     (描述最容易因为上述问题退坑的玩家特征)

#     ### 🚀 战略改进建议
#     * 🔧 **短期（1-3个月，针对研发与运营）**：(列出亟待修复的 Bug 或必须调整的广告/数值策略)
#     * 🎯 **中期（3-6个月，针对策划与发行）**：(系统玩法或商业化节奏的优化建议)
#     * 🌟 **长期（6-12个月，针对大盘生态）**：(游戏核心机制或长期留存的升级方向)

#     评论数据源:
#     {text_to_analyze}
#     """
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"AI 分析时出错: {e}"

# # ==========================================
# # 3. 网页主程序 (完全对齐 Zeus 报告结构)
# # ==========================================
# YOUR_GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
# gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# st.sidebar.header("📁 数据中心")
# uploaded_file = st.sidebar.file_uploader("上传应用市场评论数据 (CSV)", type=["csv"])

# if uploaded_file is not None:
#     df = load_and_clean_data(uploaded_file)
    
#     if df is not None:
#         st.markdown('<div class="report-header">📱 游戏 VOC 智能体检报告</div>', unsafe_allow_html=True)
#         st.caption(f"分析样本区间: {df['date'].min()} 至 {df['date'].max()} | 数据源: Google Play")
        
#         # --- 模块 1: 📊 情感分析 (对齐 Zeus 顶栏) ---
#         st.markdown('<div class="section-title">📊 情感分析与基础指标</div>', unsafe_allow_html=True)
        
#         avg_score = df['score'].mean()
#         avg_sentiment = df['nlp_sentiment_score'].mean()
#         pos_pct = (len(df[df['true_sentiment'] == 'Positive']) / len(df)) * 100
#         neg_pct = (len(df[df['true_sentiment'] == 'Negative']) / len(df)) * 100
        
#         col1, col2, col3, col4 = st.columns(4)
#         col1.metric("分析评价数", f"{len(df):,}")
#         col2.metric("平均星级评分", f"{avg_score:.2f} / 5.0")
#         col3.metric("NLP 情感均分", f"{avg_sentiment:.2f}", delta=">0为正向, <0为负向", delta_color="off")
#         col4.metric("真实负面口碑占比", f"{neg_pct:.1f}%", delta=f"好评 {pos_pct:.1f}%", delta_color="inverse")

#         # --- 模块 2: 📈 趋势分析 ---
#         st.markdown('<div class="section-title">📈 周度指标趋势分析</div>', unsafe_allow_html=True)
#         trend_col1, trend_col2 = st.columns([2, 1])
        
#         with trend_col1:
#             weekly_trend = df.groupby('week').agg(
#                 avg_score=('score', 'mean'),
#                 review_count=('score', 'count')
#             ).reset_index()
#             fig_trend = px.line(weekly_trend, x='week', y='avg_score', text=weekly_trend['avg_score'].round(2),
#                                 title="自然周平均星级走势", markers=True)
#             fig_trend.update_traces(textposition="top center")
#             st.plotly_chart(fig_trend, use_container_width=True)
            
#         with trend_col2:
#             st.markdown("<br><br>", unsafe_allow_html=True)
#             st.dataframe(weekly_trend.rename(columns={'week':'周次', 'avg_score':'平均评分', 'review_count':'评价数量'}), hide_index=True)

#         # --- 模块 3: 🤖 AI 深度洞察与战略规划 ---
#         st.markdown('<div class="section-title">🤖 AI 深度洞察与战略建议</div>', unsafe_allow_html=True)
        
#         if st.button("⚡ 立即生成专家级体检报告", type="primary"):
#             if not YOUR_GEMINI_API_KEY:
#                 st.error("未配置 API Key，无法呼叫 AI 大模型。")
#             else:
#                 with st.spinner('AI 正在交叉验证数据，生成高管级汇报纪要...'):
#                     report_content = get_zeus_style_insight(gemini_model, df)
#                     st.success("报告生成完毕")
#                     st.markdown(f"<div style='background-color: rgba(128, 128, 128, 0.1); padding: 25px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2);'>{report_content}</div>", unsafe_allow_html=True)
#                     # st.markdown(f"<div style='background-color: #ffffff; padding: 25px; border-radius: 10px; border: 1px solid #e0e0e0;'>{report_content}</div>", unsafe_allow_html=True)

#         # --- 模块 4: 词云辅助 (隐藏在折叠面板中，保持主报告清爽) ---
#         with st.expander("👁️ 查看原始文本语义聚类 (词云)"):
#             c1, c2 = st.columns(2)
#             with c1:
#                 neg_words = " ".join(df[df['true_sentiment'] == 'Negative']['content'].tolist())
#                 if neg_words:
#                     wordcloud_neg = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(re.sub(r'[^a-z\s]', '', neg_words.lower()))
#                     fig_n, ax_n = plt.subplots(figsize=(8, 4))
#                     ax_n.imshow(wordcloud_neg)
#                     ax_n.axis('off')
#                     st.pyplot(fig_n)
#             with c2:
#                 pos_words = " ".join(df[df['true_sentiment'] == 'Positive']['content'].tolist())
#                 if pos_words:
#                     wordcloud_pos = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(re.sub(r'[^a-z\s]', '', pos_words.lower()))
#                     fig_p, ax_p = plt.subplots(figsize=(8, 4))
#                     ax_p.imshow(wordcloud_pos)
#                     ax_p.axis('off')
#                     st.pyplot(fig_p)

# else:
#     st.info("👈 请在左侧边栏上传 CSV 数据以渲染分析报告。")




# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # import re
# # import google.generativeai as genai
# # import plotly.express as px
# # import os

# # # ==========================================
# # # 1. 页面级配置 & 全局 CSS 注入 (打造企业级大屏感)
# # # ==========================================
# # st.set_page_config(page_title="Bubble Shooter VOC 洞察舱", page_icon="🎯", layout="wide")

# # st.markdown("""
# # <style>
# #     .main-title {
# #         background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
# #         -webkit-background-clip: text;
# #         -webkit-text-fill-color: transparent;
# #         font-size: 3.5em;
# #         font-weight: 900;
# #         text-align: center;
# #         padding-bottom: 20px;
# #     }
# #     div[data-testid="metric-container"] {
# #         background-color: #f8f9fa;
# #         border-radius: 10px;
# #         padding: 15px;
# #         box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # st.markdown('<div class="main-title">🏹 VOC 玩家声音智能分析舱</div>', unsafe_allow_html=True)
# # st.markdown("---")

# # # ==========================================
# # # 2. 安全读取 API Key
# # # ==========================================
# # if "GEMINI_API_KEY" in st.secrets:
# #     YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# # else:
# #     YOUR_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# # # ==========================================
# # # 3. 核心功能函数
# # # ==========================================
# # def init_gemini(api_key):
# #     if not api_key: return None
# #     try:
# #         genai.configure(api_key=api_key)
# #         return genai.GenerativeModel('gemini-2.5-flash')
# #     except Exception as e:
# #         st.error(f"Gemini 初始化失败: {e}")
# #         return None

# # @st.cache_data
# # def load_and_clean_data(file_obj):
# #     try:
# #         df = pd.read_csv(file_obj)
# #         df['content'] = df['content'].fillna('')
# #         df['at'] = pd.to_datetime(df['at'])
# #         df['date'] = df['at'].dt.date # 提取日期用于趋势分析
# #         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
# #         return df
# #     except Exception as e:
# #         st.error(f"数据解析失败: {e}")
# #         return None

# # def generate_wordcloud(text, title):
# #     if not text:
# #         st.warning("没有足够的文本生成词云。")
# #         return
# #     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so', 'on', 'with', 'as', 'are', 'not', 'have'])
# #     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
# #     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
# #     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
# #     ax.imshow(wordcloud, interpolation='bilinear')
# #     ax.axis('off')
# #     st.pyplot(fig)

# # def get_ai_summary(model, reviews_list):
# #     text_to_analyze = "\n\n".join(reviews_list[:50])
# #     prompt = f"""
# #     You are an expert Game Data Analyst. Analyze these 1-2 star reviews for 'Bubble Shooter' and output a summary in Chinese.
# #     Include:
# #     1. Top 3 main complaints.
# #     2. One concrete example quote for each complaint.
# #     Reviews:
# #     {text_to_analyze}
# #     """
# #     try:
# #         return model.generate_content(prompt).text
# #     except Exception as e:
# #         return f"AI 分析时出错: {e}"

# # # ==========================================
# # # 4. 网页主程序 (交互式布局)
# # # ==========================================
# # gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# # st.sidebar.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png", use_container_width=True)
# # st.sidebar.header("📁 数据导入中心")
# # uploaded_file = st.sidebar.file_uploader("上传 Google Play CSV 评论数据", type=["csv"])

# # if not YOUR_GEMINI_API_KEY:
# #     st.sidebar.warning("⚠️ API Key 未配置，AI 分析功能受限。")

# # if uploaded_file is not None:
# #     with st.spinner('🚀 正在利用 Pandas 引擎处理数据...'):
# #         df = load_and_clean_data(uploaded_file)
    
# #     if df is not None:
# #         # --- 模块 1: 核心北极星指标 ---
# #         col1, col2, col3, col4 = st.columns(4)
# #         col1.metric("总评论数", f"{len(df):,}")
# #         col2.metric("平均得分", f"{df['score'].mean():.2f} / 5.0")
# #         col3.metric("差评占比 (1-2星)", f"{(len(df[df['sentiment'] == 'Negative']) / len(df) * 100):.1f}%")
# #         col4.metric("好评占比 (4-5星)", f"{(len(df[df['sentiment'] == 'Positive']) / len(df) * 100):.1f}%")
# #         st.markdown("<br>", unsafe_allow_html=True)

# #         # --- 模块 2: 交互式数据可视化 (Plotly) ---
# #         st.subheader("📊 玩家口碑动态分布")
# #         chart_col1, chart_col2 = st.columns([1.5, 1])
        
# #         with chart_col1:
# #             # 时间趋势图：能够看出哪一天被集中打差评
# #             daily_score = df.groupby('date')['score'].mean().reset_index()
# #             fig_line = px.line(daily_score, x='date', y='score', title='每日平均得分趋势', markers=True, color_discrete_sequence=['#FF4B2B'])
# #             st.plotly_chart(fig_line, use_container_width=True)
            
# #         with chart_col2:
# #             # 环形图：直观展示星级占比
# #             score_dist = df['score'].value_counts().reset_index()
# #             score_dist.columns = ['Score', 'Count']
# #             fig_pie = px.pie(score_dist, values='Count', names='Score', title='星级评分占比', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
# #             st.plotly_chart(fig_pie, use_container_width=True)

# #         st.markdown("---")

# #         # --- 模块 3: 词云与 AI 归因 ---
# #         st.subheader("🧠 语义洞察与 AI 诊断")
# #         tab1, tab2, tab3 = st.tabs(["✨ AI 智能差评归因", "🔴 负面词云 (痛点)", "🟢 正面词云 (爽点)"])
        
# #         with tab1:
# #             st.markdown("#### 自动生成研发团队汇报纪要")
# #             if st.button("🚀 点击唤醒 Gemini 大模型进行诊断", type="primary"):
# #                 if not YOUR_GEMINI_API_KEY:
# #                     st.error("请先在后台配置 API Key。")
# #                 else:
# #                     with st.spinner('Gemini 正在深度阅读并提炼千条差评本质...'):
# #                         neg_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
# #                         ai_summary = get_ai_summary(gemini_model, neg_reviews)
# #                         st.success("分析完成！")
# #                         st.info(ai_summary)
                        
# #         with tab2:
# #             neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
# #             generate_wordcloud(neg_text, "玩家痛点高频词")
            
# #         with tab3:
# #             pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
# #             generate_wordcloud(pos_text, "玩家爽点高频词")

# #         st.markdown("---")

# #         # --- 模块 4: 数据明细下钻 ---
# #         st.subheader("🔍 原始数据下钻")
# #         with st.expander("展开查看具体评论明细 (支持筛选与导出)"):
# #             score_filter = st.multiselect("星级过滤:", options=[5,4,3,2,1], default=[1,2,3,4,5])
# #             st.dataframe(
# #                 df[df['score'].isin(score_filter)][['at', 'score', 'content', 'reviewCreatedVersion']], 
# #                 use_container_width=True,
# #                 height=300
# #             )
# # else:
# #     st.info("👈 请在左侧边栏上传 CSV 数据，开启您的数据洞察之旅。")
# # # import streamlit as st
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # from wordcloud import WordCloud
# # # import re
# # # import google.generativeai as genai
# # # import io
# # # import os

# # # # ==========================================
# # # # 1. 配置区域 (兼容本地与云端环境)
# # # # ==========================================
# # # # st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# # # # 尝试从 Streamlit Secrets 读取（用于云端部署）
# # # if "GEMINI_API_KEY" in st.secrets:
# # #     YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# # # # 尝试从本地环境变量读取（用于本地测试）
# # # else:
# # #     YOUR_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# # # # 拦截器：如果没有读取到 Key，在页面给出友好提示
# # # if not YOUR_GEMINI_API_KEY:
# # #     st.warning("⚠️ 未检测到 Gemini API Key，AI 分析功能将受限。请在 Streamlit Cloud 的 Secrets 中完成配置。")

# # # # ==========================================
# # # # 2. 功能函数
# # # # ==========================================

# # # # 2.1 初始化 AI 模型 (Gemini)
# # # def init_gemini(api_key):
# # #     try:
# # #         if not api_key:
# # #             return None
# # #         genai.configure(api_key=api_key)
# # #         model = genai.GenerativeModel('gemini-2.5-flash')
# # #         return model
# # #     except Exception as e:
# # #         st.error(f"Gemini 初始化失败，请检查 API Key: {e}")
# # #         return None

# # # # 2.2 数据清洗与处理
# # # @st.cache_data # 缓存数据，避免每次刷新网页都重新运行
# # # def load_and_clean_data(file_obj):
# # #     # 读取你上传的 CSV 文件
# # #     try:
# # #         df = pd.read_csv(file_obj)
# # #         df['content'] = df['content'].fillna('')
# # #         df['at'] = pd.to_datetime(df['at'])
# # #         # 简单的情感打标：4-5星为正向，1-2星为负向
# # #         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
# # #         return df
# # #     except Exception as e:
# # #         st.error(f"数据读取失败: {e}")
# # #         return None

# # # # 2.3 生成词云图
# # # def generate_wordcloud(text, title):
# # #     if not text:
# # #         st.warning(f"{title} 没有足够的文本生成词云。")
# # #         return
    
# # #     # 简单的英文停用词
# # #     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
    
# # #     # 清洗文本
# # #     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
# # #     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    
# # #     # 使用 Matplotlib 显示
# # #     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
# # #     ax.imshow(wordcloud, interpolation='bilinear')
# # #     ax.axis('off')
# # #     ax.set_title(title, fontsize=20)
# # #     st.pyplot(fig)

# # # # 2.4 调用 AI (Gemini) 进行评论总结
# # # def get_ai_summary(model, reviews_list):
# # #     if not model:
# # #         return "Gemini 未初始化或 API Key 缺失。"
# # #     if not reviews_list:
# # #         return "没有足够的评论供 AI 分析。"

# # #     # 将差评论合并为一个长文本（限制长度避免 token 超限，Demo 只取前 50 条）
# # #     text_to_analyze = "\n\n".join(reviews_list[:50])
    
# # #     prompt = f"""
# # #     You are an expert Game Data Analyst and VOC (Voice of Customer) specialist. 
# # #     Below is a list of negative user reviews (1-2 stars) for our mobile game 'Bubble Shooter'.
    
# # #     Analyze these reviews and provide a summary for the development team in Chinese (中文).
# # #     Your output must include:
# # #     1. A bulleted list of the Top 3 main complaints from players.
# # #     2. For each complaint, provide one concrete example review from the text.
    
# # #     User Reviews:
# # #     \"\"\"
# # #     {text_to_analyze}
# # #     \"\"\"
# # #     """
    
# # #     try:
# # #         response = model.generate_content(prompt)
# # #         return response.text
# # #     except Exception as e:
# # #         return f"AI 分析时出错: {e}"

# # # # ==========================================
# # # # 3. 网页主界面 (Main App)
# # # # ==========================================
# # # st.title("🏹 VOC玩家声音智能分析系统")
# # # st.markdown("---")

# # # # 初始化 Gemini 模型
# # # gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# # # # 3.1 侧边栏：文件上传
# # # st.sidebar.header("数据导入")
# # # uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

# # # if uploaded_file is not None:
# # #     # 加载数据
# # #     with st.spinner('正在处理数据...'):
# # #         df = load_and_clean_data(uploaded_file)
    
# # #     if df is not None:
# # #         # ==========================================
# # #         # 模块 1: 核心评级指标 (Descriptive)
# # #         # ==========================================
# # #         st.header("1. 玩家评分概览")
# # #         col1, col2, col3 = st.columns([1, 1, 2])
        
# # #         # 指标卡
# # #         with col1:
# # #             avg_score = df['score'].mean()
# # #             st.metric(label="平均得分", value=f"{avg_score:.2f} / 5")
# # #         with col2:
# # #             total_reviews = len(df)
# # #             st.metric(label="总评论数", value=total_reviews)
            
# # #         # 评分占比图
# # #         with col3:
# # #             score_counts = df['score'].value_counts().sort_index(ascending=False)
# # #             st.bar_chart(score_counts)
        
# # #         st.markdown("---")
        
# # #         # ==========================================
# # #         # 模块 2: 核心文本洞察 (Descriptive)
# # #         # ==========================================
# # #         st.header("2. 评论文本视觉探索")
        
# # #         tab1, tab2 = st.tabs(["🔴 负面评论词云 (1-2星)", "🟢 正面评论词云 (4-5星)"])
        
# # #         with tab1:
# # #             neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
# # #             generate_wordcloud(neg_text, "Negative Reviews Keywords")
            
# # #         with tab2:
# # #             pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
# # #             generate_wordcloud(pos_text, "Positive Reviews Keywords")
            
# # #         st.markdown("---")

# # #         # ==========================================
# # #         # 模块 3: AI 智能差评归因 (Diagnostic - AI 落地)
# # #         # ==========================================
# # #         st.header("3. ✨ AI 智能差评归因 (今日 Top 3 痛点)")
        
# # #         # 准备数据供 AI 分析
# # #         negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
# # #         # 增加一个按钮触发 AI 分析，避免打开网页就自动调用扣费
# # #         if st.button("运行 AI 智能总结"):
# # #             if not YOUR_GEMINI_API_KEY:
# # #                 st.warning("请先在 Streamlit Cloud 后台的 Secrets 中填写你的 Gemini API Key。")
# # #             else:
# # #                 with st.spinner('Gemini 正在冥想并分析几百条差评...'):
# # #                     ai_summary = get_ai_summary(gemini_model, negative_reviews)
                    
# # #                     # 显示 AI 分析结果
# # #                     st.info("AI 总结报告如下：")
# # #                     st.markdown(ai_summary)
                    
# # #         # ==========================================
# # #         # 模块 4: 数据详情与筛选 (Exploratory)
# # #         # ==========================================
# # #         st.markdown("---")
# # #         st.header("4. 数据详情预览")
# # #         # 允许玩家筛选评分
# # #         score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
# # #         filtered_df = df[df['score'].isin(score_filter)]
# # #         st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion', 'sourceCountry']].head(100))

# # # else:
# # #     # 未上传文件时的欢迎界面
# # #     st.info("👋 请在侧边栏上传从 Google Play 爬取的 Bubble Shooter CSV 评论数据开始分析。")
# # #     st.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png") # 这里放置一张 Streamlit 介绍图
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt
# # # # from wordcloud import WordCloud
# # # # import re
# # # # import google.generativeai as genai
# # # # import io

# # # # # cd ~/Desktop/Bubble_VOC_Project
# # # # # streamlit run APP.py
# # # # # ==========================================
# # # # # 1. 配置区域 (需要你在本地配置)
# # # # # ==========================================
# # # # # st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# # # # # 请在此处填写你在 Google AI Studio 获取的 API Key
# # # # # 实际项目中应使用环境变量保存，这里为了 Demo 演示直接写出
# # # # YOUR_API_KEY = st.secrets["GEMINI_API_KEY"]

# # # # # ==========================================
# # # # # 2. 功能函数
# # # # # ==========================================

# # # # # 2.1 初始化 AI 模型 (Gemini)
# # # # def init_gemini(api_key):
# # # #     try:
# # # #         genai.configure(api_key=api_key)
# # # #         model = genai.GenerativeModel('gemini-2.5-flash')
# # # #         return model
# # # #     except Exception as e:
# # # #         st.error(f"Gemini 初始化失败，请检查 API Key: {e}")
# # # #         return None

# # # # # 2.2 数据清洗与处理
# # # # @st.cache_data # 缓存数据，避免每次刷新网页都重新运行
# # # # def load_and_clean_data(file_obj):
# # # #     # 读取你上传的 CSV 文件
# # # #     try:
# # # #         df = pd.read_csv(file_obj)
# # # #         df['content'] = df['content'].fillna('')
# # # #         df['at'] = pd.to_datetime(df['at'])
# # # #         # 简单的情感打标：4-5星为正向，1-2星为负向
# # # #         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
# # # #         return df
# # # #     except Exception as e:
# # # #         st.error(f"数据读取失败: {e}")
# # # #         return None

# # # # # 2.3 生成词云图
# # # # def generate_wordcloud(text, title):
# # # #     if not text:
# # # #         st.warning(f"{title} 没有足够的文本生成词云。")
# # # #         return
    
# # # #     # 简单的英文停用词
# # # #     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
    
# # # #     # 清洗文本
# # # #     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
# # # #     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    
# # # #     # 使用 Matplotlib 显示
# # # #     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
# # # #     ax.imshow(wordcloud, interpolation='bilinear')
# # # #     ax.axis('off')
# # # #     ax.set_title(title, fontsize=20)
# # # #     st.pyplot(fig)

# # # # # 2.4 调用 AI (Gemini) 进行评论总结
# # # # def get_ai_summary(model, reviews_list):
# # # #     if not model:
# # # #         return "Gemini 未初始化。"
# # # #     if not reviews_list:
# # # #         return "没有足够的评论供 AI 分析。"

# # # #     # 将差评论合并为一个长文本（限制长度避免 token 超限，Demo 只取前 50 条）
# # # #     text_to_analyze = "\n\n".join(reviews_list[:50])
    
# # # #     prompt = f"""
# # # #     You are an expert Game Data Analyst and VOC (Voice of Customer) specialist. 
# # # #     Below is a list of negative user reviews (1-2 stars) for our mobile game 'Bubble Shooter'.
    
# # # #     Analyze these reviews and provide a summary for the development team in Chinese (中文).
# # # #     Your output must include:
# # # #     1. A bulleted list of the Top 3 main complaints from players.
# # # #     2. For each complaint, provide one concrete example review from the text.
    
# # # #     User Reviews:
# # # #     \"\"\"
# # # #     {text_to_analyze}
# # # #     \"\"\"
# # # #     """
    
# # # #     try:
# # # #         response = model.generate_content(prompt)
# # # #         return response.text
# # # #     except Exception as e:
# # # #         return f"AI 分析时出错: {e}"

# # # # # ==========================================
# # # # # 3. 网页主界面 (Main App)
# # # # # ==========================================
# # # # st.title("🏹 VOC玩家声音智能分析系统")
# # # # st.markdown("---")

# # # # # 初始化 Gemini 模型
# # # # gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# # # # # 3.1 侧边栏：文件上传
# # # # st.sidebar.header("数据导入")
# # # # uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

# # # # if uploaded_file is not None:
# # # #     # 加载数据
# # # #     with st.spinner('正在处理数据...'):
# # # #         df = load_and_clean_data(uploaded_file)
    
# # # #     if df is not None:
# # # #         # ==========================================
# # # #         # 模块 1: 核心评级指标 (Descriptive)
# # # #         # ==========================================
# # # #         st.header("1. 玩家评分概览")
# # # #         col1, col2, col3 = st.columns([1, 1, 2])
        
# # # #         # 指标卡
# # # #         with col1:
# # # #             avg_score = df['score'].mean()
# # # #             st.metric(label="平均得分", value=f"{avg_score:.2f} / 5")
# # # #         with col2:
# # # #             total_reviews = len(df)
# # # #             st.metric(label="总评论数", value=total_reviews)
            
# # # #         # 评分占比图
# # # #         with col3:
# # # #             score_counts = df['score'].value_counts().sort_index(ascending=False)
# # # #             st.bar_chart(score_counts)
        
# # # #         st.markdown("---")
        
# # # #         # ==========================================
# # # #         # 模块 2: 核心文本洞察 (Descriptive)
# # # #         # ==========================================
# # # #         st.header("2. 评论文本视觉探索")
        
# # # #         tab1, tab2 = st.tabs(["🔴 负面评论词云 (1-2星)", "🟢 正面评论词云 (4-5星)"])
        
# # # #         with tab1:
# # # #             neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
# # # #             generate_wordcloud(neg_text, "Negative Reviews Keywords")
            
# # # #         with tab2:
# # # #             pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
# # # #             generate_wordcloud(pos_text, "Positive Reviews Keywords")
            
# # # #         st.markdown("---")

# # # #         # ==========================================
# # # #         # 模块 3: AI 智能差评归因 (Diagnostic - AI 落地)
# # # #         # ==========================================
# # # #         st.header("3. ✨ AI 智能差评归因 (今日 Top 3 痛点)")
        
# # # #         # 准备数据供 AI 分析
# # # #         negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
# # # #         # 增加一个按钮触发 AI 分析，避免打开网页就自动调用扣费
# # # #         if st.button("运行 AI 智能总结"):
# # # #             if YOUR_GEMINI_API_KEY == "你的_GEMINI_API_KEY_粘贴在这里":
# # # #                 st.warning("请先在 App.py 代码中填写你的 Gemini API Key。")
# # # #             else:
# # # #                 with st.spinner('Gemini 正在冥想并分析几百条差评...'):
# # # #                     ai_summary = get_ai_summary(gemini_model, negative_reviews)
                    
# # # #                     # 显示 AI 分析结果
# # # #                     st.info("AI 总结报告如下：")
# # # #                     st.markdown(ai_summary)
                    
# # # #         # ==========================================
# # # #         # 模块 4: 数据详情与筛选 (Exploratory)
# # # #         # ==========================================
# # # #         st.markdown("---")
# # # #         st.header("4. 数据详情预览")
# # # #         # 允许玩家筛选评分
# # # #         score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
# # # #         filtered_df = df[df['score'].isin(score_filter)]
# # # #         st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion', 'sourceCountry']].head(100))

# # # # else:
# # # #     # 未上传文件时的欢迎界面
# # # #     st.info("👋 请在侧边栏上传从 Google Play 爬取的 Bubble Shooter CSV 评论数据开始分析。")
# # # #     st.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png") # 这里放置一张 Streamlit 介绍图
