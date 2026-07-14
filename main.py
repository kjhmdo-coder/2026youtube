import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import urllib.request
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build

st.set_page_config(page_title="유튜브 댓글 분석기", page_icon="📺", layout="wide")

# ------------------------------
# 폰트 자동 다운로드 (한글 지원)
# ------------------------------
FONT_PATH = "NanumGothic.ttf"

@st.cache_resource
def get_font_path():
    if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 100000:
        return FONT_PATH

    urls = [
        "https://cdn.jsdelivr.net/gh/moonspam/NanumSquare@master/NanumGothic.ttf",
        "https://github.com/moonspam/NanumSquare/raw/master/NanumGothic.ttf",
        "https://raw.githubusercontent.com/googlefonts/nanum-gothic/main/fonts/ttf/NanumGothic-Regular.ttf",
    ]

    for url in urls:
        try:
            urllib.request.urlretrieve(url, FONT_PATH)
            if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 100000:
                return FONT_PATH
        except Exception:
            continue

    return None

font_path = get_font_path()

if font_path:
    try:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        pass
else:
    st.warning("⚠️ 한글 폰트 다운로드에 실패했습니다. 워드클라우드에서 한글이 깨질 수 있습니다.")

# ------------------------------
# API 키 설정
# ------------------------------
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except:
    st.error("⚠️ YOUTUBE_API_KEY가 설정되지 않았습니다. Streamlit Cloud의 Secrets에 추가해주세요.")
    st.stop()

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ------------------------------
# 유틸 함수
# ------------------------------
def extract_video_id(url_or_id):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if len(url_or_id) == 11:
        return url_or_id
    return None


def get_video_info(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    if response["items"]:
        return response["items"][0]
    return None


def get_comments(video_id, max_results=200):
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            )
            response = request.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "작성자": comment["authorDisplayName"],
                    "댓글": comment["textDisplay"],
                    "좋아요": comment["likeCount"],
                    "작성일": comment["publishedAt"],
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            st.warning(f"댓글을 가져오는 중 오류가 발생했습니다: {e}")
            break

    return comments


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_stopwords():
    return set([
        "이", "그", "저", "것", "수", "등", "들", "및", "를", "에", "의", "가", "은", "는",
        "이다", "있다", "하다", "되다", "않다", "없다", "같다", "너무", "정말", "진짜",
        "그리고", "그런데", "하지만", "그래서", "때문", "이런", "저런", "그런",
        "제가", "저는", "나는", "우리", "당신", "여기", "거기", "저기", "이것", "그것",
        "저것", "무엇", "어떤", "어떻게", "왜", "어디", "누가", "언제", "은근", "진짜",
        "완전", "그냥", "약간", "조금", "매우", "아주", "더", "덜", "또", "또한", "역시",
        "video", "channel", "youtube", "http", "https", "com", "www"
    ])


def simple_sentiment_analysis(text):
    positive_words = ["좋다", "최고", "감사", "사랑", "멋지다", "훌륭", "대박", "굿", "good", 
                       "great", "love", "thank", "best", "amazing", "재밌", "웃기", "행복",
                       "예쁘", "잘하", "짱", "굳", "완벽", "감동"]
    negative_words = ["싫다", "별로", "최악", "나쁘", "실망", "화나", "짜증", "bad", "worst",
                       "hate", "terrible", "disappointing", "구리", "노잼", "지루", "별로",
                       "아쉽", "부족", "실패"]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return "긍정"
    elif neg_count > pos_count:
        return "부정"
    else:
        return "중립"


# ------------------------------
# 메인 앱
# ------------------------------
st.title("📺 유튜브 댓글 분석기")
st.markdown("유튜브 영상의 댓글을 분석하고 멋진 워드클라우드를 만들어보세요!")

with st.sidebar:
    st.header("⚙️ 설정")
    video_input = st.text_input(
        "유튜브 URL 또는 영상 ID",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    max_comments = st.slider("가져올 댓글 수", 50, 500, 200, step=50)
    analyze_button = st.button("🔍 분석 시작", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 📖 사용 방법")
    st.markdown("""
    1. 유튜브 영상 URL을 입력하세요
    2. 가져올 댓글 수를 설정하세요
    3. '분석 시작' 버튼을 클릭하세요
    """)

if analyze_button and video_input:
    video_id = extract_video_id(video_input)

    if not video_id:
        st.error("올바른 유튜브 URL 또는 영상 ID를 입력해주세요.")
    else:
        with st.spinner("영상 정보를 가져오는 중..."):
            video_info = get_video_info(video_id)

        if not video_info:
            st.error("영상 정보를 찾을 수 없습니다. URL을 확인해주세요.")
        else:
            snippet = video_info["snippet"]
            stats = video_info["statistics"]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(snippet["thumbnails"]["high"]["url"], use_container_width=True)
            with col2:
                st.subheader(snippet["title"])
                st.caption(f"채널: {snippet['channelTitle']}")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("조회수", f"{int(stats.get('viewCount', 0)):,}")
                with m2:
                    st.metric("좋아요", f"{int(stats.get('likeCount', 0)):,}")
                with m3:
                    st.metric("댓글 수", f"{int(stats.get('commentCount', 0)):,}")

            st.markdown("---")

            with st.spinner(f"댓글 {max_comments}개를 가져오는 중..."):
                comments = get_comments(video_id, max_comments)

            if not comments:
                st.warning("댓글을 가져올 수 없습니다. 댓글이 없거나 비활성화된 영상일 수 있습니다.")
            else:
                df = pd.DataFrame(comments)
                df["작성일"] = pd.to_datetime(df["작성일"])
                df["정제댓글"] = df["댓글"].apply(clean_text)
                df["감성"] = df["댓글"].apply(simple_sentiment_analysis)

                st.success(f"✅ 총 {len(df)}개의 댓글을 성공적으로 가져왔습니다!")

                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    ["☁️ 워드클라우드", "😊 감성 분석", "🏆 인기 댓글", "📊 통계", "📋 전체 댓글"]
                )

                # ------------------------------
                # 워드클라우드 탭
                # ------------------------------
                with tab1:
                    st.subheader("☁️ 댓글 워드클라우드")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        colormap = st.selectbox(
                            "색상 테마",
                            ["viridis", "plasma", "inferno", "magma", "cividis", 
                             "cool", "spring", "summer", "autumn", "winter", "rainbow"]
                        )
                        bg_color = st.color_picker("배경 색상", "#FFFFFF")
                        max_words = st.slider("최대 단어 수", 50, 300, 150)
                        shape = st.selectbox("모양", ["기본(사각형)", "원형"])

                    all_text = " ".join(df["정제댓글"].tolist())
                    words = all_text.split()
                    stopwords = get_stopwords()
                    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
                    word_freq = Counter(filtered_words)

                    if word_freq and font_path:
                        mask = None
                        if shape == "원형":
                            x, y = np.ogrid[:600, :600]
                            mask_circle = (x - 300) ** 2 + (y - 300) ** 2 > 290 ** 2
                            mask = 255 * mask_circle.astype(int)

                        try:
                            wc = WordCloud(
                                font_path=font_path,
                                width=1200,
                                height=800,
                                background_color=bg_color,
                                colormap=colormap,
                                max_words=max_words,
                                mask=mask,
                                relative_scaling=0.5,
                                min_font_size=10,
                                prefer_horizontal=0.9
                            ).generate_from_frequencies(word_freq)

                            fig, ax = plt.subplots(figsize=(14, 9))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)

                            import io as io_module
                            buf = io_module.BytesIO()
                            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", 
                                       facecolor=bg_color)
                            st.download_button(
                                label="📥 워드클라우드 다운로드",
                                data=buf.getvalue(),
                                file_name=f"wordcloud_{video_id}.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"워드클라우드 생성 오류: {e}")
                    elif not font_path:
                        st.error("한글 폰트를 사용할 수 없어 워드클라우드를 생성할 수 없습니다.")
                    else:
                        st.warning("분석할 단어가 충분하지 않습니다.")

                    st.markdown("---")
                    st.subheader("🔝 최다 언급 단어 TOP 20")
                    top_words = word_freq.most_common(20)
                    if top_words:
                        word_df = pd.DataFrame(top_words, columns=["단어", "빈도수"])
                        fig_bar = px.bar(
                            word_df, x="빈도수", y="단어", orientation="h",
                            color="빈도수", color_continuous_scale=colormap,
                            title="단어 빈도수 TOP 20"
                        )
                        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_bar, use_container_width=True)

                # ------------------------------
                # 감성 분석 탭
                # ------------------------------
                with tab2:
                    st.subheader("😊 댓글 감성 분석")

                    sentiment_counts = df["감성"].value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="감성 분포",
                            color=sentiment_counts.index,
                            color_discrete_map={"긍정": "#2ecc71", "부정": "#e74c3c", "중립": "#95a5a6"}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col2:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("😊 긍정", sentiment_counts.get("긍정", 0))
                        with m2:
                            st.metric("😐 중립", sentiment_counts.get("중립", 0))
                        with m3:
                            st.metric("😞 부정", sentiment_counts.get("부정", 0))

                        st.markdown("### 감성별 예시 댓글")
                        selected_sentiment = st.selectbox("감성 선택", ["긍정", "중립", "부정"])
                        sample_comments = df[df["감성"] == selected_sentiment].nlargest(5, "좋아요")
                        for _, row in sample_comments.iterrows():
                            st.info(f"**{row['작성자']}**: {row['댓글'][:100]}")

                # ------------------------------
                # 인기 댓글 탭
                # ------------------------------
                with tab3:
                    st.subheader("🏆 좋아요 TOP 10 댓글")

                    top_comments = df.nlargest(10, "좋아요")

                    for idx, (_, row) in enumerate(top_comments.iterrows(), 1):
                        with st.container():
                            col1, col2 = st.columns([1, 10])
                            with col1:
                                st.markdown(f"### #{idx}")
                            with col2:
                                st.markdown(f"**{row['작성자']}** · 👍 {row['좋아요']}")
                                st.write(row["댓글"])
                                st.caption(f"감성: {row['감성']}")
                            st.markdown("---")

                # ------------------------------
                # 통계 탭
                # ------------------------------
                with tab4:
                    st.subheader("📊 댓글 통계")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("전체 댓글 수", len(df))
                    with col2:
                        st.metric("평균 좋아요", f"{df['좋아요'].mean():.1f}")
                    with col3:
                        st.metric("최다 좋아요", df["좋아요"].max())
                    with col4:
                        avg_length = df["댓글"].str.len().mean()
                        st.metric("평균 댓글 길이", f"{avg_length:.0f}자")

                    st.markdown("---")

                    st.subheader("📅 댓글 작성 시간 트렌드")
                    df["날짜"] = df["작성일"].dt.date
                    daily_counts = df.groupby("날짜").size().reset_index(name="댓글수")

                    fig_line = px.line(
                        daily_counts, x="날짜", y="댓글수",
                        title="일별 댓글 작성 추이", markers=True
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                    st.subheader("📏 댓글 길이 분포")
                    df["댓글길이"] = df["댓글"].str.len()
                    fig_hist = px.histogram(
                        df, x="댓글길이", nbins=30,
                        title="댓글 길이 히스토그램"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.subheader("👍 좋아요 수 분포")
                    fig_box = px.box(df, y="좋아요", title="좋아요 수 박스플롯")
                    st.plotly_chart(fig_box, use_container_width=True)

                # ------------------------------
                # 전체 댓글 탭
                # ------------------------------
                with tab5:
                    st.subheader("📋 전체 댓글 목록")

                    sort_option = st.selectbox("정렬 기준", ["좋아요 많은 순", "최신순", "오래된 순"])

                    display_df = df[["작성자", "댓글", "좋아요", "작성일", "감성"]].copy()

                    if sort_option == "좋아요 많은 순":
                        display_df = display_df.sort_values("좋아요", ascending=False)
                    elif sort_option == "최신순":
                        display_df = display_df.sort_values("작성일", ascending=False)
                    else:
                        display_df = display_df.sort_values("작성일", ascending=True)

                    st.dataframe(display_df, use_container_width=True, height=500)

                    csv = display_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 CSV로 다운로드",
                        data=csv,
                        file_name=f"youtube_comments_{video_id}.csv",
                        mime="text/csv"
                    )

elif analyze_button and not video_input:
    st.warning("유튜브 URL 또는 영상 ID를 입력해주세요.")

else:
    st.info("👈 왼쪽 사이드바에서 유튜브 URL을 입력하고 분석을 시작해보세요!")
    
    st.markdown("""
    ### ✨ 주요 기능
    - **워드클라우드**: 댓글에서 자주 등장하는 단어를 멋지게 시각화
    - **감성 분석**: 댓글의 긍정/부정/중립 비율 분석
    - **인기 댓글**: 좋아요가 많은 댓글 TOP 10
    - **통계 분석**: 댓글 작성 트렌드, 길이 분포 등
    - **전체 댓글**: 모든 댓글을 정렬하고 CSV로 다운로드
    """)
