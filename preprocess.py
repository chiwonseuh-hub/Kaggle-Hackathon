import pandas as pd
import json
import os

# --- 설정 (사용자 환경에 맞게 수정) ---
CSV_PATH = 'dataset/ddi_data/ddi_metadata.csv'        # DDI 메타데이터 경로
IMAGE_DIR = 'dataset/ddi_data/images/'       # DDI 이미지 폴더 경로
OUTPUT_FILE = 'dataset/ddi_data/train_data.jsonl' # 출력 파일명

def create_medical_instruction(row):
    """
    CSV 한 줄을 Gemma 4 학습용 대화 포맷으로 변환합니다.
    """
    # CSV 컬럼명에 맞춰 수정 (예: 'label', 'skin_tone', 'image_path')
    label = row.get('label', '알 수 없는 병변')
    skin_tone = row.get('skin_tone', '알 수 없음')
    img_filename = row.get('image_path', '') # 혹은 'DDI_0001.jpg' 등
    
    # 모델에게 시킬 작업(Instruction) 정의
    user_prompt = f"이 피부 이미지의 병변을 분석하고 진단명을 추론해줘. (피부톤: {skin_tone})"
    
    # 모델이 내놓아야 할 정답(Assistant Response) 정의
    assistant_response = f"이미지 분석 결과, 이 병변은 '{label}'(으)로 추론됩니다. 피부톤 {skin_tone} 단계에서의 전형적인 임상 특징을 보입니다."

    # Gemma 4 Conversational Format (Unsloth 권장 포맷)
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image"}]}, # 이미지가 포함됨을 명시
            {"role": "assistant", "content": [{"type": "text", "text": assistant_response}]}
        ],
        "image": os.path.join(IMAGE_DIR, img_filename) # 실제 이미지 경로
    }

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ 에러: {CSV_PATH} 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"📂 총 {len(df)}개의 데이터를 불러왔습니다.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            example = create_medical_instruction(row)
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✅ 전처리 완료! 생성된 파일: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()