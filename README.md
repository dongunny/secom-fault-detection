# 🏭 SECOM 반도체 공정 불량 예측 AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**반도체 공정 센서 데이터 기반 머신러닝 불량 탐지 시스템**
BER **29.7%** 달성 — 논문(McCann & Johnston, 2008) 대비 **3.8%p 개선**

## 핵심 성과

| 지표 | 본 연구 (LR) | 논문 최고 기준 |
|------|------------|-------------|
| **BER (낮을수록 좋음)** | **29.7% ✅** | 33.5% |
| **AUC (높을수록 좋음)** | 0.756 | — |
| **불량 감지율 (TPR)** | **65.4%** | 59.1% |
| **양품 정확도 (TNR)** | 75.1% | 73.8% |

## 데이터셋

SECOM Dataset — UCI Machine Learning Repository
- 샘플 수: 1,567개
- 변수 수: 591개 공정 센서
- 불량률: 6.6% (극심한 불균형)

## 실행 방법

```bash
pip install -r requirements.txt
python analysis.py    # ML 파이프라인 실행
python gen_visual.py  # 시각화 생성
```

## 방법론

원본 591개 → 결측률>50% 제거(-28) → 분산=0 제거(-116) → 중앙값대치 → StandardScaler → F-test Top 40 → 10-fold CV → LR/RF 비교

## 참고문헌

McCann, M., & Johnston, A. (2008). SECOM Dataset. UCI Machine Learning Repository.
