<div align="center">

# SECOM 반도체 공정 불량 예측 AI

**반도체 공정 센서 데이터 기반 머신러닝 불량 탐지 시스템**

> BER **29.7%** 달성 — McCann & Johnston (2008) 논문 최고치(33.5%) 대비 **3.8%p 개선**

</div>

---

## 📌 프로젝트 개요

반도체 제조 공정(SECOM)에서 수집된 **1,567개 샘플 × 591개 센서 변수** 데이터를 분석하여, 웨이퍼 불량(Fail)을 공정 중 조기에 자동 탐지하는 AI 분류 시스템을 구축했습니다.

### 🏆 핵심 성과

| 지표 | 본 연구 (LR) | 논문 최고 기준 (F-test) | 개선폭 |
|------|------------|---------------------|-------|
| **BER ↓** | **29.7%** ✅ | 33.5% | **3.8%p** |
| **AUC ↑** | 0.756 | — | — |
| **불량 감지율 (TPR) ↑** | **65.4%** | 59.1% | 6.3%p |
| **양품 정확도 (TNR) ↑** | 75.1% | 73.8% | 1.3%p |

---

## 🎯 문제 정의

반도체 공장에서 수백 개의 센서가 동시에 공정 데이터를 기록합니다. 이 센서 신호를 실시간 분석해 **불량 웨이퍼를 공정 완료 전에 조기 탐지**하면 후속 공정 낭비를 막고 수율을 높일 수 있습니다.

**핵심 도전 과제:**
- 591개 변수 중 실제로 불량과 연관된 변수 식별
- **93.4% Pass / 6.6% Fail** 극심한 클래스 불균형 처리
- 최대 91.2%에 달하는 결측값 처리 전략
- 논문 수준의 Balanced Error Rate(BER) 달성

---

## 📊 데이터셋

**SECOM Dataset** — UCI Machine Learning Repository (McCann & Johnston, 2008)

```
샘플 수     : 1,567개   (반도체 제품)
변수 수     :   591개   (공정 센서 측정값)
양품(Pass)  : 1,463개  (93.4%)
불량(Fail)  :   104개  ( 6.6%)
평균 결측률 :   4.54%
최대 결측률 :  91.2%
클래스 비율 :  14.1 : 1  ← 극심한 불균형!
```

> ⚠️ **왜 Accuracy가 아닌 BER을 쓰나?**  
> 모든 샘플을 Pass로 예측해도 Accuracy = 93.4%가 나옵니다. 이는 불량을 전혀 잡지 못하는 쓸모없는 모델입니다.  
> **BER(Balanced Error Rate) = (오탐율 + 누락율) / 2** — 낮을수록 좋음

---

## 🔬 방법론

### 전체 파이프라인

```
원본 데이터 (1,567 × 591개 변수)
    │
    ▼ [Step 1] 결측률 > 50% 변수 제거 → 28개 삭제
    │
    ▼ [Step 2] 분산 = 0 변수 제거 → 116개 삭제
    │           (남은 변수: 447개)
    ▼ [Step 3] 중앙값 대치 + StandardScaler 정규화
    │
    ▼ [Step 4] F-test(ANOVA) → 상위 40개 특징 선택
    │
    ▼ [Step 5] 10-fold Stratified Cross-Validation
    │
    ├── Logistic Regression (C=0.1, balanced)  → BER 29.7% ✅
    └── Random Forest (200 trees, depth=8)      → BER 47.0%
    │
    ▼ [평가] BER / AUC / TPR / TNR
```

---

## 📈 주요 결과

### 모델 성능 비교

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🥇 Logistic Regression  ← 최종 선택 모델
  ┌──────────────────────────────────────────────┐
  │  BER       : 29.7%   ← 논문(33.5%) 대비 3.8%p ↓  │
  │  AUC       : 0.756                              │
  │  불량 감지율 : 65.4%  (104개 중 68개 정확 탐지)   │
  │  양품 정확도 : 75.1%                             │
  └──────────────────────────────────────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 혼동 행렬 (Logistic Regression)

```
                  예측: Pass    예측: Fail
실제: Pass(1,463)   1,099 ✅       364 ⚠️
실제: Fail  (104)      36 ⚠️        68 ✅

  TN = 1,099  │  FP = 364  │  FN = 36  │  TP = 68
```

---

## 📁 디렉토리 구조

```
secom-fault-detection/
├── analysis.py               # 🔬 전체 ML 파이프라인 (핵심 코드)
├── gen_visual.py             # 📊 한글 고품질 시각화 생성기
├── dashboard.html            # 📈 인터랙티브 결과 대시보드
├── SECOM_Analysis_Colab.ipynb # ☁️ Google Colab 실행 노트북
├── requirements.txt          # 📦 패키지 목록
├── README.md                 # 📄 (이 파일)
│
├── results/
│   ├── results.json          # 모든 수치 결과 (JSON)
│   └── fig1~fig6.png         # 기본 matplotlib 차트
│
└── results_visual/           # 🎨 한글 고품질 시각화 (6종)
    ├── 01_데이터_개요.png
    ├── 02_모델_성능_게이지.png
    ├── 03_ROC_BER_비교.png
    ├── 04_혼동행렬.png
    ├── 05_공정변수_중요도.png
    └── 06_파이프라인_흐름도.png
```

---

## ⚡ 빠른 시작

### 옵션 A — Google Colab (권장, 설치 불필요)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hBWqAF1jHysknxeF2IJTpd6E0OMvDuyH)

### 옵션 B — 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/dongunny/secom-fault-detection.git
cd secom-fault-detection

# 2. 가상환경 생성 및 패키지 설치
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

---

<div align="center">

**🏭 2026 반도체 FDC AI 프로젝트**  
Made with ❤️ | SECOM Fault Detection

</div>
