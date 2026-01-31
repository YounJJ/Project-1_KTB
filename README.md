# Project-1_KTB
# ML-Driven Interest Rate Forecasting and Bond Performance Attribution System

## 1. Project Overview
This project is an **End-to-End Risk Management System** built to bridge the gap between theoretical modeling and practical trading needs. It integrates **Machine Learning (XGBoost)** with traditional **Financial Engineering (NSS Model)** to forecast Korea Treasury Bond (KTB) yields and analyze portfolio performance under simulated scenarios.

Unlike static risk measures, this system acts as an **Early Warning System** for Asset-Liability Management (ALM). It doesn't just predict rates; it quantifies the exact financial impact (P/L) of those predictions on a bond portfolio, helping risk managers make data-driven decisions before the market moves.

## 2. Key Features & Methodology
The pipeline consists of 5 phases, designed to mimic a real-world quantitative workflow:

### Phase 1: Data Engineering & Curve Construction
- **Goal:** Transform raw market data into a usable term structure.
- **Method:** Fetched historical yields (2017-2025) via Oracle DB and applied the **Nelson-Siegel-Svensson (NSS)** model to construct a continuous Yield Curve, ensuring smooth calibration across all tenors.

### Phase 2: ML Forecasting (Tenor-specific Modeling)
- **Goal:** Predict the shape of the yield curve for the next 6 months.
- **Method:** Trained **Decoupled XGBoost Models** for key tenors (3Y, 10Y, 50Y). This approach captures distinct market dynamics—monetary policy impacts on short-term bonds vs. macro-fundamentals on long-term bonds.
- **Validation:** Utilized **Walk-forward Validation** to confirm robustness, successfully predicting the upward trend (Bear Market) in H2 2025.

### Phase 3: Valuation (Pricing Engine)
- **Goal:** Calculate fair value based on the predicted curve.
- **Method:** Built a custom `BondPricing` engine. It applies the ML-forecasted yield shifts to current market levels to derive the expected bond prices for year-end 2025.

### Phase 4: Performance Attribution (Reporting)
- **Goal:** Understand *why* we made or lost money.
- **Method:** Decomposed Total P/L into three drivers using a Waterfall Analysis:
  1.  **Carry & Roll-down:** Deterministic returns from time passing.
  2.  **Rate Change:** P/L driven by the shift in the benchmark curve.
  3.  **Spread Effect:** Residual P/L from liquidity or credit premiums.
- **Output:** Generates detailed **Attribution Tables** and **Waterfall Charts** for clear reporting.

### Phase 5: Risk Management (ALM Stress Test Dashboard)
- **Goal:** Assess solvency under the predicted stress scenario.
- **Method:** Conducted a **Duration Gap Analysis** based on the ML-predicted "Most Likely Scenario".
- **Output:** Produces a comprehensive **ALM Stress Test Dashboard** that visualizes changes in **Total Assets, Liabilities, and Net Asset Value (NAV)**, providing actionable insights for capital buffering.

---

## 3. Financial Implications

### "Quantifying Risk with High Fidelity"
Backtesting results for H2 2025 showed that the system is highly effective at quantifying downside risk in a rising rate environment.

- **Accurate Trend Prediction:** The ML model correctly forecasted the **Bear Market (Yield Rise)**, signaling significant valuation losses ahead of time.
- **P/L Attribution Accuracy:**
  - **Expected P/L (Model):** -306.99 (10Y), -1,119.71 (50Y)
  - **Actual P/L (Market):** -325.39 (10Y), -1,127.12 (50Y)
  - **Insight:** The model explained over **94% of the Actual P/L**. This suggests that the losses were primarily driven by the **Rate Change (Delta)**—which the model predicted—rather than unpredictable spread volatility.
- **Value:** The system empowers managers to proactively adjust **Duration** or implement **Hedges** by providing concrete estimates of potential capital erosion.

---

## 4. Dataset Description
*(Raw data excluded for compliance)*
- **Market Data:** Daily KTB Yields (2017.01 ~ 2025.12) sourced from BOK (ECOS).
- **Bond Specs:**
  - 10Y Benchmark: KTB 02625-3506 (Mat: Jun 2035)
  - 50Y Benchmark: KTB 02750-7409 (Mat: Sep 2074)

---

## 5. Directory Structure
```text
ML-Driven-Bond-Attribution-System/
├── yield_curve_model/            # Core Logic & Analytics Scripts
│   ├── models.py                 # Libraries: NSS, XGBoost Forecast, BondPricing, Attribution
│   ├── data_loader.py            # Oracle DB Data Handler
│   ├── run_nss_validation.py     # Step 1: Curve Fitting Check
│   ├── run_forecast_validation.py# Step 2: Forecast Accuracy Check
│   ├── run_expected_vs_actual.py # Step 3: P/L Simulation (Generates Table & Curve plot)
│   ├── run_attribution.py        # Step 4: Attribution Analysis (Generates Waterfall & Table)
│   └── run_stress_test.py        # Step 5: ALM Stress Test (Generates Dashboard)
│
├── images/                       # Generated Visual Reports
│   ├── nss_validation.png
│   ├── forecast_validation.png
│   ├── expected_vs_actual_curve.png
│   ├── expected_vs_actual_table.png # [New] P/L Comparison Table
│   ├── attribution_waterfall.png
│   ├── attribution_table.png        # [New] Detailed Decomposition Table
│   └── stress_test_dashboard.png    # [New] ALM Risk Dashboard
│
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```
------

# 머신러닝 기반 금리 예측 및 채권 성과 분해 시스템
**(ML-Driven Interest Rate Forecasting and Bond Performance Attribution System)**

## 1. 프로젝트 소개
이 프로젝트는 국고채 금리 예측에 **머신러닝(XGBoost)** 기법을 도입하고, 이를 전통적인 금융공학 모델(**NSS Curve Fitting**)과 결합하여 채권 포트폴리오의 미래 변화를 시뮬레이션하는 **End-to-End 리스크 관리 시스템**입니다.

단순히 금리의 등락(Up/Down)을 맞추는 것을 넘어, 예측된 수익률 곡선(Yield Curve)이 내 포트폴리오의 **평가 손익(Valuation P/L)에 미칠 구체적인 금액**을 산출함으로써, 리스크 매니저가 시장 변동 전에 선제적으로 대응할 수 있도록 돕는 **조기 경보 시스템(Early Warning System)** 구축을 목표로 했습니다.

## 2. 주요 기능 및 방법론 (Key Features)
실무적인 퀀트 분석 프로세스를 반영하여 총 5단계의 파이프라인으로 구성했습니다.

### Phase 1: 데이터 엔지니어링 & 커브 구성
- **목표:** 원천 데이터를 분석 가능한 형태의 이자율 기간구조로 변환
- **방법:** Oracle DB에서 2017~2025년 국고채 금리를 수집하고, **Nelson-Siegel-Svensson (NSS)** 모델을 적용해 전 만기에 걸친 연속적인 **수익률 곡선(Yield Curve)** 을 생성했습니다.

### Phase 2: ML 예측 (만기별 모델링)
- **목표:** 향후 6개월 뒤의 수익률 곡선(Term Structure) 형상 예측
- **방법:** 3년(단기), 10년/50년(장기) 등 만기별로 특화된 **XGBoost 모델(Decoupled Models)** 을 구축하여, 통화정책(단기)과 거시경제(장기) 요인을 각각 반영했습니다.
- **검증:** **Walk-forward Validation**을 수행하여 2025년 하반기 금리 상승(Bear Market) 추세를 성공적으로 포착했습니다.

### Phase 3: 가치 평가 (Pricing Engine)
- **목표:** 예측된 금리를 기반으로 개별 채권의 이론가 산출
- **방법:** 자체 개발한 `BondPricing` 엔진을 통해, ML 모델이 예측한 **금리 변동분(Delta)** 을 현재 시장가에 반영하여 연말 시점의 기대 가격을 도출했습니다.

### Phase 4: 성과 요인 분해 (Reporting)
- **목표:** 손익이 '어디서' 발생했는지 원인 규명 (Attribution Analysis)
- **방법:** 워터폴(Waterfall) 분석을 통해 전체 손익을 3가지 핵심 요소로 분해했습니다.
  1.  **Carry & Roll-down:** 시간 경과에 따른 이자 수익 및 롤다운 효과 (확정 이익)
  2.  **Rate Change:** 금리 변동에 따른 자본 손익 (모델 예측 영역)
  3.  **Spread Effect:** 유동성 및 크레딧 등 모델 외적인 잔여 손익
- **결과물:** 직관적인 **성과 분석 테이블(`attribution_table`)** 과 **워터폴 차트(`attribution_waterfall`)** 를 자동 생성하여 보고서 형태로 출력합니다.

### Phase 5: 리스크 관리 (ALM Stress Test Dashboard)
- **목표:** 예측된 시나리오 하에서 조직의 건전성(Solvency) 평가
- **방법:** ML 예측 시나리오를 Stress Case로 설정하고, 자산-부채의 **듀레이션 갭(Duration Gap)** 분석을 수행했습니다.
- **결과물:** **자산(Asset), 부채(Liability), 순자산(NAV)** 의 변동을 한눈에 파악할 수 있는 **ALM 스트레스 테스트 대시보드(`stress_test_dashboard`)** 를 생성합니다.

---

## 3. 분석 결과 및 의미 (Financial Implications)

### "리스크의 정량화 (Quantifying Risk)"
2025년 하반기를 대상으로 한 백테스팅 결과, 본 시스템은 금리 상승기(Bear Market)의 잠재 손실을 매우 정교하게 예측했습니다.

- **추세 예측 적중:** ML 모델은 금리의 **우상향 추세**를 정확히 짚어냈으며, 이에 따른 채권 평가 손실을 사전에 경고했습니다.
- **높은 설명력:**
  - **예상 손익 (Model):** -306.99 (10Y), -1,119.71 (50Y)
  - **실제 손익 (Actual):** -325.39 (10Y), -1,127.12 (50Y)
  - **인사이트:** 모델의 예상 손익이 실제 손익의 **94% 이상을 설명**했습니다. 이는 실제 발생한 손실의 대부분이 모델이 예측한 **금리 요인(Rate Change)** 에 기인했다는 것을 의미하며, Spread 등 예측 불가능한 잡음(Noise)의 영향은 제한적이었음을 보여줍니다.
- **활용 가치:** 결과적으로 이 시스템은 미래의 자본 감소분을 구체적인 수치로 제시함으로써, **듀레이션 축소**나 **헤지 포지션** 구축과 같은 의사결정에 확실한 근거를 제공합니다.

---

## 4. 데이터셋 설명
*(보안상 원본 데이터 파일은 저장소에 포함되지 않았습니다)*
- **Market Data:** 2017.01 ~ 2025.12 국고채 일별 금리 (출처: 한국은행 ECOS)
- **Bond Specs:**
  - 10년 지표물: 국고02625-3506 (만기: 2035-06)
  - 50년 지표물: 국고02750-7409 (만기: 2074-09)

---

## 5. 디렉토리 구조 (Directory Structure)

```text
ML-Driven-Bond-Attribution-System/
├── yield_curve_model/            # 핵심 로직 및 실행 스크립트
│   ├── models.py                 # 모델 클래스 (NSS, XGBoost, Pricing, Attribution)
│   ├── data_loader.py            # Oracle DB 데이터 로더
│   ├── run_nss_validation.py     # [1단계] 커브 적합도 검증
│   ├── run_forecast_validation.py# [2단계] 금리 예측 모델 검증
│   ├── run_expected_vs_actual.py # [3단계] 예상 손익 시뮬레이션 (Table & Curve 생성)
│   ├── run_attribution.py        # [4단계] 성과 요인 분해 (Waterfall & Table 생성)
│   └── run_stress_test.py        # [5단계] ALM 스트레스 테스트 (Dashboard 생성)
│
├── images/                       # 결과 리포트 및 시각화
│   ├── nss_validation.png
│   ├── forecast_validation.png
│   ├── expected_vs_actual_curve.png
│   ├── expected_vs_actual_table.png # [New] 손익 비교 테이블
│   ├── attribution_waterfall.png
│   ├── attribution_table.png        # [New] 성과 분해 상세 테이블
│   └── stress_test_dashboard.png    # [New] ALM 리스크 대시보드
│
├── requirements.txt              # 사용된 패키지
└── README.md                     # 프로젝트 문서
