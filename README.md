<div align="center">

# 🏭 SECOM 반도체 공정 불량 예측 AI
### Semi-Conductor Manufacturing Fault Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_SECOM-7c3aed?style=for-the-badge)](https://archive.ics.uci.edu/dataset/179/secom)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br>

**반도체 공정 센서 데이터 기반 머신러닝 불량 탐지 시스템**

> BER **29.7%** 달성 — 논문(McCann & Johnston, 2008) **33.5%** 대비 **3.8%p 개선** 🏆

</div>

---

## 🎯 프로젝트 한 줄 요약

> UCI SECOM 반도체 공정 데이터셋 **(1,567샘플 × 591변수)** 을 분석하여, 극심한 클래스 불균형(불량률 6.6%) 문제를 해결하고 **논문 수준을 뛰어넘는 불량 탐지 모델**을 구축한 프로젝트입니다.

---

## 📊 핵심 성과 요약

| 모델 | BER ↓ | AUC ↑ | 불량 감지율 (TPR) | 양품 정확도 (TNR) | 평가 |
|------|-------|-------|----------------|----------------|------|
| **Logistic Regression** | **29.7%** | 0.756 | **65.4%** | 75.1% | 🏆 Best BER |
| Random Forest | 47.0% | **0.768** | 7.7% | **98.2%** | Best AUC |
| S2N (논문 기준) | 34.5% | — | — | — | Reference |
| T-test (논문 기준) | 33.7% | — | — | — | Reference |
| **F-test (논문 최고)** | **33.5%** | — | 59.1% | 73.8% | Reference |

> 💡 **BER (Balanced Error Rate)** = (오탐율 + 누락율) / 2 — 불균형 데이터에서 가장 공정한 평가 지표

---

## 🔬 문제 정의 및 연구 배경

### 왜 반도체 FDC(Fault Detection & Classification)인가?

반도체 제조 공정에서 **하나의 웨이퍼 가격은 수백만 원**에 달합니다. 불량 웨이퍼를 후공정에서 발견할수록 손실은 기하급수적으로 증가합니다.

```
[공정 입력] ──→ [센서 모니터링] ──→ [AI 불량 탐지] ──→ [공정 즉시 개선]
   원재료          591개 센서             ML 모델           수율 향상
```

### SECOM 데이터셋이란?

**McCann & Johnston (2008)** 논문에서 공개된 실제 반도체 제조 공정 데이터입니다.

| 항목 | 값 |
|------|-----|
| 출처 | UCI Machine Learning Repository |
| 샘플 수 | **1,567개** (반도체 제품) |
| 변수 수 | **591개** (공정 센서 측정값) |
| 양품(Pass) | **1,463개 (93.4%)** |
| 불량(Fail) | **104개 (6.6%)** |
| 클래스 비율 | **14.1 : 1** (극심한 불균형) |
| 평균 결측률 | 4.54% |
| 최대 결측률 | **91.2%** |

### 핵심 도전과제

1. **극심한 클래스 불균형** — 불량 14.1배 희귀 → 단순 모델은 모든 샘플을 Pass로 예측
2. **고차원 희소 데이터** — 591개 변수, 최대 91% 결측률
3. **익명화된 변수명** — F1~F591로만 표시, 도메인 지식 적용 불가
4. **BER 최소화** — 단순 정확도가 아닌 균형 잡힌 오류율 최소화

---

## 🛠️ 기술 스택

| 구분 | 기술 | 버전 | 용도 |
|------|------|------|------|
| 언어 | Python | 3.10+ | 전체 파이프라인 |
| ML | scikit-learn | 1.8.0 | 모델·전처리·평가 |
| 데이터 | pandas | 3.0.2 | 데이터 처리 |
| 수치 | numpy | 2.4.4 | 행렬 연산 |
| 시각화 | matplotlib | 3.10.9 | 차트 생성 |
| 시각화 | seaborn | 0.13.2 | 히트맵 |
| 통계 | scipy | 1.17.1 | F-test |

---

## 📁 디렉토리 구조

```
secom-fault-detection/
│
├── 📄 README.md                      ← 이 파일
├── 📄 requirements.txt               ← 패키지 의존성
├── 📄 LICENSE                        ← MIT 라이선스
│
├── 🔬 analysis.py                    ← 전체 ML 파이프라인
├── 📊 gen_visual.py                  ← 한글 시각화 생성기 (6종)
├── 🌐 dashboard.html                 ← 인터랙티브 결과 대시보드
│
├── results/
│   ├── 📋 results.json               ← 모든 수치 결과
│   ├── fig1_eda_overview.png         ← EDA 개요
│   ├── fig2_missing_heatmap.png      ← 결측값 히트맵
│   ├── fig3_feature_importance.png   ← F-test 중요도
│   ├── fig4_model_performance.png    ← 모델 성능
│   ├── fig5_confusion_matrix.png     ← 혼동 행렬
│   └── fig6_rf_importance.png        ← RF 중요도
│
└── results_visual/                   ← 고품질 한글 시각화
    ├── 01_데이터_개요.png
    ├── 02_모델_성능_게이지.png
    ├── 03_ROC_BER_비교.png
    ├── 04_혼동행렬.png
    ├── 05_공정변수_중요도.png
    └── 06_파이프라인_흐름도.png
```

---

## ⚡ 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/dongunny/secom-fault-detection.git
cd secom-fault-detection
```

### 2. 가상환경 & 패키지 설치

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 데이터 다운로드

```bash
mkdir secom
cd secom
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data" -OutFile "secom.data"
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data" -OutFile "secom_labels.data"
cd ..
```

### 4. 분석 실행

```bash
# Step 1: 전체 ML 파이프라인 실행 (약 2~3분)
python analysis.py

# Step 2: 고품질 시각화 생성
python gen_visual.py

# Step 3: 대시보드 열기
start dashboard.html    # Windows
```

---

## 🔬 방법론 상세

### 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECOM ML 파이프라인                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  원본 데이터          전처리                  특징 선택              │
│  1,567 × 591   →   결측률>50% 제거(-28)  →  F-test ANOVA        │
│                →   분산=0 제거(-116)     →  Top 40 선택           │
│                →   중앙값 대치            →  1,567 × 40           │
│                →   StandardScaler                                 │
│                                                                   │
│  10-fold Stratified CV                                            │
│  ├── Logistic Regression (C=0.1, balanced) → BER 29.7% ✅        │
│  └── Random Forest (200 trees, balanced)   → BER 47.0%           │
│                                                                   │
│  평가: BER · AUC · TPR · TNR · Confusion Matrix                   │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| 결측값 처리 | 중앙값 대치 | 반도체 센서 이상치에 강건 |
| 특징 선택 | F-test Top 40 | 통계적 유의성 + 차원 축소 |
| 불균형 처리 | `class_weight='balanced'` | Fail에 자동으로 ~14배 가중치 |
| 교차검증 | Stratified 10-fold | fold별 불량 비율 일정 유지 |
| 평가 지표 | BER | 불균형 데이터 공정 평가 |
| LR 정규화 | C=0.1 | 고차원에서 과적합 방지 |

### 왜 Logistic Regression이 Random Forest보다 좋은가?

```
Random Forest의 역설:
  AUC = 0.768 > LR의 0.756  (확률 추정은 RF가 더 우수)
  BER = 47.0% >> LR의 29.7% (실제 분류 성능은 LR이 훨씬 우수)

원인: RF가 임계값 0.5에서 대부분 Pass로 예측
      → TPR = 7.7% (104개 중 8개만 탐지)
      → BER 급등

해결: 임계값 최적화 또는 SMOTE 오버샘플링
```

---

## 📈 상세 결과

### 혼동 행렬 — Logistic Regression (Best BER)

```
                    예측: Pass      예측: Fail
실제: Pass (1,463)   1,099 (TN✅)     364 (FP⚠️)
실제: Fail  (104)       36 (FN⚠️)      68 (TP✅)

TN = 1,099  →  양품을 양품으로 정확히 분류 (75.1%)
FP =   364  →  양품을 불량으로 과잉 판단  (24.9%)
FN =    36  →  불량을 양품으로 놓침 ⚠️    (34.6%)
TP =    68  →  불량을 불량으로 정확히 탐지 (65.4%)
```

> ⚠️ **FN 최소화가 핵심!** 불량이 고객에게 출하되는 FN이 가장 위험합니다.
> `class_weight='balanced'` 없이는 FN ≈ 104개(전부 놓침)가 됩니다.

### 핵심 공정 변수 Top 10 (F-test 기준)

| 순위 | 변수 | F-통계량 | 의미 |
|------|------|---------|------|
| 1 | **F59** | 39.04 | 불량/양품 간 가장 큰 통계적 차이 |
| 2 | **F103** | 36.63 | 두 번째로 강한 불량 신호 |
| 3 | **F510** | 27.61 | 세 번째 핵심 공정 변수 |
| 4 | **F348** | 27.24 | 네 번째 주요 센서 |
| 5 | **F431** | 22.84 | 다섯 번째 |
| 6 | **F434** | 19.63 | 여섯 번째 |
| 7 | **F430** | 18.86 | 일곱 번째 |
| 8 | **F21** | 18.59 | 여덟 번째 |
| 9 | **F435** | 18.56 | 아홉 번째 |
| 10 | **F28** | 18.12 | 열 번째 |

---

## 🚀 향후 개선 계획

- [ ] **SMOTE 오버샘플링** — Fail 샘플을 합성으로 늘려 불균형 근본 해결
- [ ] **XGBoost / LightGBM** — 그래디언트 부스팅으로 BER 20%대 목표
- [ ] **임계값 최적화** — Precision-Recall 곡선으로 최적 임계값 탐색
- [ ] **SHAP 분석** — 개별 예측 설명 (XAI)
- [ ] **AutoEncoder** — 비지도 학습 기반 이상 탐지
- [ ] **앙상블** — Voting / Stacking으로 모델 결합

---

## 🧪 Colab에서 바로 실행

Google Colab에서 설치 없이 실행 가능한 노트북을 제공합니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

노트북 구성 (21개 셀):
- 환경 설정 → UCI 데이터 자동 다운로드 → EDA → 전처리 → F-test → 모델 학습 → 평가 → 시각화 → 결론

---

## 📚 참고문헌

```bibtex
@misc{secom2008,
  author       = {McCann, M. and Johnston, A.},
  title        = {{SECOM} {D}ataset},
  year         = {2008},
  howpublished = {{UCI} {M}achine {L}earning {R}epository},
  url          = {https://archive.ics.uci.edu/dataset/179/secom},
  note         = {DOI: 10.24432/C5VW2N}
}

@book{ESL2009,
  author    = {Hastie, T. and Tibshirani, R. and Friedman, J.},
  title     = {The Elements of Statistical Learning},
  edition   = {2nd},
  year      = {2009},
  publisher = {Springer},
  address   = {New York}
}

@article{Breiman2001,
  author  = {Breiman, L.},
  title   = {Random {F}orests},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001}
}
```

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

<div align="center">

**2026 반도체 FDC AI 프로젝트**

Made with ❤️ by dongunny

⭐ Star this repo if it helped you!

</div>
