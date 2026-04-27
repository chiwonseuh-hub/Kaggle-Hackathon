import streamlit as st
from ollama import generate, chat
from PIL import Image
import io

# --- UI 설정 ---
st.set_page_config(page_title="Gemma 4 Medical Aid", layout="wide")
st.title("🩺 Gemma 4 Local Medical Assistant")
st.info("이 서비스는 로컬에서 작동하며, 입력된 데이터는 외부로 전송되지 않습니다.")

# --- 사이드바: 설정 및 가이드 ---
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select Model", ["gemma4:E2B"])
    st.divider()
    st.write("📌 **사용 가이드**")
    st.write("1. 증상을 입력하거나 환부 사진을 업로드하세요.")
    st.write("2. Gemma 4가 의학 가이드라인을 기반으로 분석합니다.")

# --- 메인 인터페이스 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Section")
    user_input = st.text_area("증상을 상세히 입력해 주세요:", placeholder="예: 어제부터 팔에 붉은 반점이 생기고 가려워요.")
    uploaded_file = st.file_uploader("환부 사진 업로드 (DDI 데이터셋 스타일)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)

system_prompt = """
당신은 전문적인 의료 분석 보조 AI입니다. 
제공된 이미지와 증상을 바탕으로 다음 단계에 따라 분석을 진행하세요:
1. 시각적 특징 기술: 피부 병변의 색상, 모양, 크기, 분포 등을 객관적으로 설명하세요.
2. 가이드라인 대조: 증상을 일반적인 의학적 징후와 비교하세요.
3. 주의 사항: 현재 상황에서 즉시 조치해야 할 사항이나 피해야 할 행동을 조언하세요.
4. 권고: 어떤 진료과를 방문해야 하는지 추천하세요.

주의: 당신은 의사가 아닙니다. 답변 끝에 반드시 '이 분석은 참고용이며 확진이 아닙니다'라는 경고 문구를 포함하세요.
"""

with col2:
    st.subheader("Gemma 4 Analysis (Thinking Mode)")
    
    if st.button("분석 시작"):
        if user_input or uploaded_file:
            with st.spinner("Gemma가 분석 중입니다..."):
                # 이미지 처리 (Ollama API 규격)
                images = []
                if uploaded_file:
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    images = [buffered.getvalue()]

                # Ollama를 통한 Gemma 4 호출
                try:
                    response = chat(
                model="gemma4:e2b",
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"증상: {user_input}", 'images': images}
                ]
            )
                    
                    st.markdown("### 📋 분석 결과")
                    st.write(response['message']['content'])
                    
                    st.warning("⚠️ 본 결과는 AI의 추론이며, 반드시 전문의와 상담하십시오.")
                except Exception as e:
                    st.error(f"Error: {e}\nOllama가 실행 중인지 확인하세요.")
        else:
            st.warning("텍스트를 입력하거나 이미지를 업로드해 주세요.")