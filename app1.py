
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import google.generativeai as genai
import plotly.express as px
import os

# ==========================================
# 1. 页面级配置 & 全局 CSS 注入 (打造企业级大屏感)
# ==========================================
st.set_page_config(page_title="Bubble Shooter VOC 洞察舱", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main-title {
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5em;
        font-weight: 900;
        text-align: center;
        padding-bottom: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏹 VOC 玩家声音智能分析舱</div>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. 安全读取 API Key
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    YOUR_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ==========================================
# 3. 核心功能函数
# ==========================================
def init_gemini(api_key):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Gemini 初始化失败: {e}")
        return None

@st.cache_data
def load_and_clean_data(file_obj):
    try:
        df = pd.read_csv(file_obj)
        df['content'] = df['content'].fillna('')
        df['at'] = pd.to_datetime(df['at'])
        df['date'] = df['at'].dt.date # 提取日期用于趋势分析
        df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
        return df
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None

def generate_wordcloud(text, title):
    if not text:
        st.warning("没有足够的文本生成词云。")
        return
    stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so', 'on', 'with', 'as', 'are', 'not', 'have'])
    clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def get_ai_summary(model, reviews_list):
    text_to_analyze = "\n\n".join(reviews_list[:50])
    prompt = f"""
    You are an expert Game Data Analyst. Analyze these 1-2 star reviews for 'Bubble Shooter' and output a summary in Chinese.
    Include:
    1. Top 3 main complaints.
    2. One concrete example quote for each complaint.
    Reviews:
    {text_to_analyze}
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"AI 分析时出错: {e}"

# ==========================================
# 4. 网页主程序 (交互式布局)
# ==========================================
gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

st.sidebar.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png", use_container_width=True)
st.sidebar.header("📁 数据导入中心")
uploaded_file = st.sidebar.file_uploader("上传 Google Play CSV 评论数据", type=["csv"])

if not YOUR_GEMINI_API_KEY:
    st.sidebar.warning("⚠️ API Key 未配置，AI 分析功能受限。")

if uploaded_file is not None:
    with st.spinner('🚀 正在利用 Pandas 引擎处理数据...'):
        df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        # --- 模块 1: 核心北极星指标 ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总评论数", f"{len(df):,}")
        col2.metric("平均得分", f"{df['score'].mean():.2f} / 5.0")
        col3.metric("差评占比 (1-2星)", f"{(len(df[df['sentiment'] == 'Negative']) / len(df) * 100):.1f}%")
        col4.metric("好评占比 (4-5星)", f"{(len(df[df['sentiment'] == 'Positive']) / len(df) * 100):.1f}%")
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 模块 2: 交互式数据可视化 (Plotly) ---
        st.subheader("📊 玩家口碑动态分布")
        chart_col1, chart_col2 = st.columns([1.5, 1])
        
        with chart_col1:
            # 时间趋势图：能够看出哪一天被集中打差评
            daily_score = df.groupby('date')['score'].mean().reset_index()
            fig_line = px.line(daily_score, x='date', y='score', title='每日平均得分趋势', markers=True, color_discrete_sequence=['#FF4B2B'])
            st.plotly_chart(fig_line, use_container_width=True)
            
        with chart_col2:
            # 环形图：直观展示星级占比
            score_dist = df['score'].value_counts().reset_index()
            score_dist.columns = ['Score', 'Count']
            fig_pie = px.pie(score_dist, values='Count', names='Score', title='星级评分占比', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # --- 模块 3: 词云与 AI 归因 ---
        st.subheader("🧠 语义洞察与 AI 诊断")
        tab1, tab2, tab3 = st.tabs(["✨ AI 智能差评归因", "🔴 负面词云 (痛点)", "🟢 正面词云 (爽点)"])
        
        with tab1:
            st.markdown("#### 自动生成研发团队汇报纪要")
            if st.button("🚀 点击唤醒 Gemini 大模型进行诊断", type="primary"):
                if not YOUR_GEMINI_API_KEY:
                    st.error("请先在后台配置 API Key。")
                else:
                    with st.spinner('Gemini 正在深度阅读并提炼千条差评本质...'):
                        neg_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
                        ai_summary = get_ai_summary(gemini_model, neg_reviews)
                        st.success("分析完成！")
                        st.info(ai_summary)
                        
        with tab2:
            neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
            generate_wordcloud(neg_text, "玩家痛点高频词")
            
        with tab3:
            pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
            generate_wordcloud(pos_text, "玩家爽点高频词")

        st.markdown("---")

        # --- 模块 4: 数据明细下钻 ---
        st.subheader("🔍 原始数据下钻")
        with st.expander("展开查看具体评论明细 (支持筛选与导出)"):
            score_filter = st.multiselect("星级过滤:", options=[5,4,3,2,1], default=[1,2,3,4,5])
            st.dataframe(
                df[df['score'].isin(score_filter)][['at', 'score', 'content', 'reviewCreatedVersion']], 
                use_container_width=True,
                height=300
            )
else:
    st.info("👈 请在左侧边栏上传 CSV 数据，开启您的数据洞察之旅。")
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import re
# import google.generativeai as genai
# import io
# import os

# # ==========================================
# # 1. 配置区域 (兼容本地与云端环境)
# # ==========================================
# # st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# # 尝试从 Streamlit Secrets 读取（用于云端部署）
# if "GEMINI_API_KEY" in st.secrets:
#     YOUR_GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# # 尝试从本地环境变量读取（用于本地测试）
# else:
#     YOUR_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# # 拦截器：如果没有读取到 Key，在页面给出友好提示
# if not YOUR_GEMINI_API_KEY:
#     st.warning("⚠️ 未检测到 Gemini API Key，AI 分析功能将受限。请在 Streamlit Cloud 的 Secrets 中完成配置。")

# # ==========================================
# # 2. 功能函数
# # ==========================================

# # 2.1 初始化 AI 模型 (Gemini)
# def init_gemini(api_key):
#     try:
#         if not api_key:
#             return None
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel('gemini-2.5-flash')
#         return model
#     except Exception as e:
#         st.error(f"Gemini 初始化失败，请检查 API Key: {e}")
#         return None

# # 2.2 数据清洗与处理
# @st.cache_data # 缓存数据，避免每次刷新网页都重新运行
# def load_and_clean_data(file_obj):
#     # 读取你上传的 CSV 文件
#     try:
#         df = pd.read_csv(file_obj)
#         df['content'] = df['content'].fillna('')
#         df['at'] = pd.to_datetime(df['at'])
#         # 简单的情感打标：4-5星为正向，1-2星为负向
#         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
#         return df
#     except Exception as e:
#         st.error(f"数据读取失败: {e}")
#         return None

# # 2.3 生成词云图
# def generate_wordcloud(text, title):
#     if not text:
#         st.warning(f"{title} 没有足够的文本生成词云。")
#         return
    
#     # 简单的英文停用词
#     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
    
#     # 清洗文本
#     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
#     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    
#     # 使用 Matplotlib 显示
#     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     ax.set_title(title, fontsize=20)
#     st.pyplot(fig)

# # 2.4 调用 AI (Gemini) 进行评论总结
# def get_ai_summary(model, reviews_list):
#     if not model:
#         return "Gemini 未初始化或 API Key 缺失。"
#     if not reviews_list:
#         return "没有足够的评论供 AI 分析。"

#     # 将差评论合并为一个长文本（限制长度避免 token 超限，Demo 只取前 50 条）
#     text_to_analyze = "\n\n".join(reviews_list[:50])
    
#     prompt = f"""
#     You are an expert Game Data Analyst and VOC (Voice of Customer) specialist. 
#     Below is a list of negative user reviews (1-2 stars) for our mobile game 'Bubble Shooter'.
    
#     Analyze these reviews and provide a summary for the development team in Chinese (中文).
#     Your output must include:
#     1. A bulleted list of the Top 3 main complaints from players.
#     2. For each complaint, provide one concrete example review from the text.
    
#     User Reviews:
#     \"\"\"
#     {text_to_analyze}
#     \"\"\"
#     """
    
#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"AI 分析时出错: {e}"

# # ==========================================
# # 3. 网页主界面 (Main App)
# # ==========================================
# st.title("🏹 VOC玩家声音智能分析系统")
# st.markdown("---")

# # 初始化 Gemini 模型
# gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# # 3.1 侧边栏：文件上传
# st.sidebar.header("数据导入")
# uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

# if uploaded_file is not None:
#     # 加载数据
#     with st.spinner('正在处理数据...'):
#         df = load_and_clean_data(uploaded_file)
    
#     if df is not None:
#         # ==========================================
#         # 模块 1: 核心评级指标 (Descriptive)
#         # ==========================================
#         st.header("1. 玩家评分概览")
#         col1, col2, col3 = st.columns([1, 1, 2])
        
#         # 指标卡
#         with col1:
#             avg_score = df['score'].mean()
#             st.metric(label="平均得分", value=f"{avg_score:.2f} / 5")
#         with col2:
#             total_reviews = len(df)
#             st.metric(label="总评论数", value=total_reviews)
            
#         # 评分占比图
#         with col3:
#             score_counts = df['score'].value_counts().sort_index(ascending=False)
#             st.bar_chart(score_counts)
        
#         st.markdown("---")
        
#         # ==========================================
#         # 模块 2: 核心文本洞察 (Descriptive)
#         # ==========================================
#         st.header("2. 评论文本视觉探索")
        
#         tab1, tab2 = st.tabs(["🔴 负面评论词云 (1-2星)", "🟢 正面评论词云 (4-5星)"])
        
#         with tab1:
#             neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
#             generate_wordcloud(neg_text, "Negative Reviews Keywords")
            
#         with tab2:
#             pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
#             generate_wordcloud(pos_text, "Positive Reviews Keywords")
            
#         st.markdown("---")

#         # ==========================================
#         # 模块 3: AI 智能差评归因 (Diagnostic - AI 落地)
#         # ==========================================
#         st.header("3. ✨ AI 智能差评归因 (今日 Top 3 痛点)")
        
#         # 准备数据供 AI 分析
#         negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
#         # 增加一个按钮触发 AI 分析，避免打开网页就自动调用扣费
#         if st.button("运行 AI 智能总结"):
#             if not YOUR_GEMINI_API_KEY:
#                 st.warning("请先在 Streamlit Cloud 后台的 Secrets 中填写你的 Gemini API Key。")
#             else:
#                 with st.spinner('Gemini 正在冥想并分析几百条差评...'):
#                     ai_summary = get_ai_summary(gemini_model, negative_reviews)
                    
#                     # 显示 AI 分析结果
#                     st.info("AI 总结报告如下：")
#                     st.markdown(ai_summary)
                    
#         # ==========================================
#         # 模块 4: 数据详情与筛选 (Exploratory)
#         # ==========================================
#         st.markdown("---")
#         st.header("4. 数据详情预览")
#         # 允许玩家筛选评分
#         score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
#         filtered_df = df[df['score'].isin(score_filter)]
#         st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion', 'sourceCountry']].head(100))

# else:
#     # 未上传文件时的欢迎界面
#     st.info("👋 请在侧边栏上传从 Google Play 爬取的 Bubble Shooter CSV 评论数据开始分析。")
#     st.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png") # 这里放置一张 Streamlit 介绍图
# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from wordcloud import WordCloud
# # import re
# # import google.generativeai as genai
# # import io

# # # cd ~/Desktop/Bubble_VOC_Project
# # # streamlit run APP.py
# # # ==========================================
# # # 1. 配置区域 (需要你在本地配置)
# # # ==========================================
# # # st.set_page_config(page_title="Bubble Shooter VOC Dashboard", layout="wide")

# # # 请在此处填写你在 Google AI Studio 获取的 API Key
# # # 实际项目中应使用环境变量保存，这里为了 Demo 演示直接写出
# # YOUR_API_KEY = st.secrets["GEMINI_API_KEY"]

# # # ==========================================
# # # 2. 功能函数
# # # ==========================================

# # # 2.1 初始化 AI 模型 (Gemini)
# # def init_gemini(api_key):
# #     try:
# #         genai.configure(api_key=api_key)
# #         model = genai.GenerativeModel('gemini-2.5-flash')
# #         return model
# #     except Exception as e:
# #         st.error(f"Gemini 初始化失败，请检查 API Key: {e}")
# #         return None

# # # 2.2 数据清洗与处理
# # @st.cache_data # 缓存数据，避免每次刷新网页都重新运行
# # def load_and_clean_data(file_obj):
# #     # 读取你上传的 CSV 文件
# #     try:
# #         df = pd.read_csv(file_obj)
# #         df['content'] = df['content'].fillna('')
# #         df['at'] = pd.to_datetime(df['at'])
# #         # 简单的情感打标：4-5星为正向，1-2星为负向
# #         df['sentiment'] = df['score'].map({5: 'Positive', 4: 'Positive', 3: 'Neutral', 2: 'Negative', 1: 'Negative'})
# #         return df
# #     except Exception as e:
# #         st.error(f"数据读取失败: {e}")
# #         return None

# # # 2.3 生成词云图
# # def generate_wordcloud(text, title):
# #     if not text:
# #         st.warning(f"{title} 没有足够的文本生成词云。")
# #         return
    
# #     # 简单的英文停用词
# #     stop_words = set(['the', 'and', 'to', 'i', 'a', 'it', 'is', 'of', 'this', 'you', 'for', 'in', 'that', 'game', 'but', 'my', 'play', 'so'])
    
# #     # 清洗文本
# #     clean_text = re.sub(r'[^a-z\s]', '', text.lower())
    
# #     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='Dark2').generate(clean_text)
    
# #     # 使用 Matplotlib 显示
# #     fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
# #     ax.imshow(wordcloud, interpolation='bilinear')
# #     ax.axis('off')
# #     ax.set_title(title, fontsize=20)
# #     st.pyplot(fig)

# # # 2.4 调用 AI (Gemini) 进行评论总结
# # def get_ai_summary(model, reviews_list):
# #     if not model:
# #         return "Gemini 未初始化。"
# #     if not reviews_list:
# #         return "没有足够的评论供 AI 分析。"

# #     # 将差评论合并为一个长文本（限制长度避免 token 超限，Demo 只取前 50 条）
# #     text_to_analyze = "\n\n".join(reviews_list[:50])
    
# #     prompt = f"""
# #     You are an expert Game Data Analyst and VOC (Voice of Customer) specialist. 
# #     Below is a list of negative user reviews (1-2 stars) for our mobile game 'Bubble Shooter'.
    
# #     Analyze these reviews and provide a summary for the development team in Chinese (中文).
# #     Your output must include:
# #     1. A bulleted list of the Top 3 main complaints from players.
# #     2. For each complaint, provide one concrete example review from the text.
    
# #     User Reviews:
# #     \"\"\"
# #     {text_to_analyze}
# #     \"\"\"
# #     """
    
# #     try:
# #         response = model.generate_content(prompt)
# #         return response.text
# #     except Exception as e:
# #         return f"AI 分析时出错: {e}"

# # # ==========================================
# # # 3. 网页主界面 (Main App)
# # # ==========================================
# # st.title("🏹 VOC玩家声音智能分析系统")
# # st.markdown("---")

# # # 初始化 Gemini 模型
# # gemini_model = init_gemini(YOUR_GEMINI_API_KEY)

# # # 3.1 侧边栏：文件上传
# # st.sidebar.header("数据导入")
# # uploaded_file = st.sidebar.file_uploader("上传爬取的 CSV 评论数据", type=["csv"])

# # if uploaded_file is not None:
# #     # 加载数据
# #     with st.spinner('正在处理数据...'):
# #         df = load_and_clean_data(uploaded_file)
    
# #     if df is not None:
# #         # ==========================================
# #         # 模块 1: 核心评级指标 (Descriptive)
# #         # ==========================================
# #         st.header("1. 玩家评分概览")
# #         col1, col2, col3 = st.columns([1, 1, 2])
        
# #         # 指标卡
# #         with col1:
# #             avg_score = df['score'].mean()
# #             st.metric(label="平均得分", value=f"{avg_score:.2f} / 5")
# #         with col2:
# #             total_reviews = len(df)
# #             st.metric(label="总评论数", value=total_reviews)
            
# #         # 评分占比图
# #         with col3:
# #             score_counts = df['score'].value_counts().sort_index(ascending=False)
# #             st.bar_chart(score_counts)
        
# #         st.markdown("---")
        
# #         # ==========================================
# #         # 模块 2: 核心文本洞察 (Descriptive)
# #         # ==========================================
# #         st.header("2. 评论文本视觉探索")
        
# #         tab1, tab2 = st.tabs(["🔴 负面评论词云 (1-2星)", "🟢 正面评论词云 (4-5星)"])
        
# #         with tab1:
# #             neg_text = " ".join(df[df['sentiment'] == 'Negative']['content'].tolist())
# #             generate_wordcloud(neg_text, "Negative Reviews Keywords")
            
# #         with tab2:
# #             pos_text = " ".join(df[df['sentiment'] == 'Positive']['content'].tolist())
# #             generate_wordcloud(pos_text, "Positive Reviews Keywords")
            
# #         st.markdown("---")

# #         # ==========================================
# #         # 模块 3: AI 智能差评归因 (Diagnostic - AI 落地)
# #         # ==========================================
# #         st.header("3. ✨ AI 智能差评归因 (今日 Top 3 痛点)")
        
# #         # 准备数据供 AI 分析
# #         negative_reviews = df[df['sentiment'] == 'Negative']['content'].tolist()
        
# #         # 增加一个按钮触发 AI 分析，避免打开网页就自动调用扣费
# #         if st.button("运行 AI 智能总结"):
# #             if YOUR_GEMINI_API_KEY == "你的_GEMINI_API_KEY_粘贴在这里":
# #                 st.warning("请先在 App.py 代码中填写你的 Gemini API Key。")
# #             else:
# #                 with st.spinner('Gemini 正在冥想并分析几百条差评...'):
# #                     ai_summary = get_ai_summary(gemini_model, negative_reviews)
                    
# #                     # 显示 AI 分析结果
# #                     st.info("AI 总结报告如下：")
# #                     st.markdown(ai_summary)
                    
# #         # ==========================================
# #         # 模块 4: 数据详情与筛选 (Exploratory)
# #         # ==========================================
# #         st.markdown("---")
# #         st.header("4. 数据详情预览")
# #         # 允许玩家筛选评分
# #         score_filter = st.multiselect("筛选星级:", options=[5,4,3,2,1], default=[1,2])
# #         filtered_df = df[df['score'].isin(score_filter)]
# #         st.dataframe(filtered_df[['at', 'score', 'content', 'reviewCreatedVersion', 'sourceCountry']].head(100))

# # else:
# #     # 未上传文件时的欢迎界面
# #     st.info("👋 请在侧边栏上传从 Google Play 爬取的 Bubble Shooter CSV 评论数据开始分析。")
# #     st.image("https://global.discourse-cdn.com/business7/uploads/streamlit/original/3X/8/0/805e3f421115f5c3897103d15b1356e9c9160565.png") # 这里放置一张 Streamlit 介绍图
