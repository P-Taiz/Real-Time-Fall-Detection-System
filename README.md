# Real-Time Fall Detection System - 실시간 AI 기반 낙상 감지 시스템

## 프로젝트 개요
본 프로젝트는 YOLO과 MediaPipe를 활용하여 실시간으로 사람의 낙상을 감지하고, Telegram을 통해 즉각적인 경고를 전송하는 시스템을 개발하는 것을 목표로 합니다. 이 시스템은 병원, 요양 시설, 가정 등 다양한 환경에서 낙상 사고를 예방하고 신속히 대응할 수 있도록 설계되었습니다. 주요 기능은 다음과 같습니다:
- **사람 감지**: YOLOv8을 이용해 이미지 또는 비디오에서 사람을 정확히 탐지.
- **자세 추정**: MediaPipe를 통해 신체의 주요 랜드마크를 분석하여 자세를 추정.
- **낙상 감지**: 신체 비율(aspect ratio)과 기울기 각도(body angle)를 기반으로 낙상을 판단.
- **실시간 경고**: 낙상이 감지되면 Telegram Bot을 통해 즉각적인 알림 전송.

## 프로젝트 배경
노인 인구의 증가와 함께 낙상 사고는 심각한 사회적 문제로 대두되고 있습니다. 특히 고령자는 낙상으로 인해 심각한 부상을 입을 가능성이 높으며, 이는 빠른 대응이 필요합니다. 본 시스템은 실시간으로 낙상을 감지하고 경고를 전송하여 사고로 인한 피해를 최소화하고자 합니다.

## 시스템 구조
시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다:
1. **YOLO**: 사람 탐지를 위해 사용되며, 신뢰도(confidence) 0.7 이상인 경우에만 탐지된 객체를 처리.
2. **MediaPipe**: 33개의 신체 랜드마크를 추출하여 자세를 분석.
3. **낙상 감지 알고리즘**: Aspect Ratio, 신체 기울기 각도, Sliding Window 기법을 활용해 낙상을 판단.
4. **Telegram 경고 시스템**: 낙상이 감지되면 JPEG 형식의 이미지와 함께 알림을 전송.
5. **Streamlit UI**: 실시간 비디오 스트리밍과 낙상 감지 결과를 시각적으로 표시.

## 설치 및 실행 방법
### 요구 사항
- Python 3.8 이상
- 프로젝트에 필요한 모든 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다.

### 설치
1. 리포지토리를 클론합니다:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
3. YOLO 모델 가중치 파일(`weights/best.pt`)을 다운로드하여 `weights` 폴더에 저장합니다.
4. Telegram Bot 토큰과 Chat ID를 `telegram_alert.py`에 설정합니다:
   ```python
   BOT_TOKEN = 'your_bot_token'
   CHAT_ID = 'your_chat_id'
   ```

### 실행
1. 기본 실행 (OpenCV 창으로 표시):
   ```bash
   python main.py
   ```
2. Streamlit UI로 실행:
   ```bash
   streamlit run app.py
   ```

## 주요 코드 파일
- **`main.py`**: YOLO과 MediaPipe를 통합하여 실시간 낙상 감지를 수행하는 메인 스크립트.
- **`fall_detector.py`**: MediaPipe 기반 자세 추정 및 낙상 감지 알고리즘을 구현.
- **`telegram_alert.py`**: Telegram Bot API를 통해 낙상 알림을 전송.
- **`app.py`**: Streamlit을 이용한 웹 기반 사용자 인터페이스.

## 시각화 자료
### 이미지 예시
![낙상 감지 결과](Test\Test1.png)
*설명*: Streamlit UI에서 표시된 실시간 낙상 감지 결과. 빨간색 바운딩 박스는 낙상을 나타내며, 초록색은 정상 상태를 나타냅니다.

### 비디오 데모
[낙상 감지 데모 비디오](Test\Example1.mp4)
*설명*: 시스템이 실시간으로 낙상을 감지하는는 과정을 보여줍니다.

## 참고 자료
1. 고령자의 낙상 예방을 위한 정책 및 프로그램  
   https://www.korea.kr/briefing/pressReleaseView.do?newsId=156638189#pressRelease
2. Merck Manuals, Falls in Older Adults  
   https://www.merckmanuals.com/professional/geriatrics/falls-in-older-adults/falls-in-older-adults
3. Salimi, M. et al., Using Deep Neural Networks for Human Fall Detection Based on Pose Estimation, Sensors, 2022.  
   https://www.mdpi.com/1424-8220/22/12/4544#fig_body_display_sensors-22-04544-f001
4. Ultralytics, "YOLOv8 by Ultralytics," Ultralytics, 2023.  
   https://docs.ultralytics.com
5. Google MediaPipe 공식 문서  
   https://ai.google.dev/edge/mediapipe/solutions
6. Yong Chen, Weitong Li, Lu Wang, Jiajia Hu., Vision Based Fall Event Detection in Complex Background Using Attention Guided Bi-Directional LSTM  
   https://www.researchgate.net/publication/346894576_Vision-Based_Fall_Event_Detection_in_Complex_Background_Using_Attention_Guided_Bi-Directional_LSTM
7. Telegram Bot API 공식 문서  
   https://core.telegram.org/bots/api
8. Mandatory Service, Falling detection using OpenCV and Mediapipe for Industry  
   https://github.com/onenationonemind1/falling_defiance
9. Code With Aarohi, YOLO-NAS Custom Object Detection  
   https://youtu.be/pgf9bPuEsF0?sj=MR7y7VPJFWNPCVbZ
10. Ali Zara, Muhammad Haroon Yousaf, Waqar Ahmad, Sergio A. Velastin, Serestina Viriri, Human fall detection using pose estimation: From traditional machine learning to vision transformers.  
    https://youtu.be/pgf9bPuEsF0?sj=oPJoRV9-PTeICn_1