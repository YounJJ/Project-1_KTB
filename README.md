# Project-1_KTB
# ML-Driven Interest Rate Forecasting and Bond Performance Attribution System

## 1. Project Overview
This project is an **End-to-End Risk Management System** that integrates Machine Learning (XGBoost) with traditional financial engineering models (NSS Curve Fitting) to forecast Korea Treasury Bond (KTB) yields and analyze bond portfolio performance.

The system is designed to overcome the limitations of static risk measures by predicting future **Yield Curves** and quantifying potential **P/L (Profit and Loss)**. It successfully demonstrates the ability to act as an **Early Warning System** for Asset-Liability Management (ALM).

## 2. Key Features & Methodology
The project pipeline consists of 5 distinct phases:

### Phase 1: Data Engineering & Curve Construction
- **Source:** Historical Yields (2017-2025) via Oracle DB.
- **Method:** Utilized the **Nelson-Siegel-Svensson (NSS)** model to convert discrete term structure data into continuous **Yield Curves**, ensuring smooth calibration across all tenors.

### Phase 2: ML Forecasting (Tenor-specific Modeling)
- **Model:** XGBoost Regressor.
- **Approach:** Implemented **Decoupled Models** for specific tenors (3Y, 10Y, 50Y) to capture distinct market dynamics—monetary policy sensitivity for short-term and macro-fundamental factors for long-term bonds.
- **Validation:** Validated using **Walk-forward Analysis**, confirming an upward trend in yields (Bear Market) for H2 2025.

### Phase 3: Valuation (Pricing Engine)
- **Engine:** Developed a custom `BondPricing` class for precise cash flow discounting.
- **Logic:** Applied ML-predicted **yield differences** to current levels to construct the forecasted term structure for year-end 2025.

### Phase 4: Performance Attribution (Waterfall Analysis)
Decomposed Total P/L into three core components to verify the source of returns:
1.  **Carry & Roll-down:** Deterministic returns from time decay.
2.  **Rate Change:** Capital P/L driven by benchmark curve shifts.
3.  **Spread Effect:** Residual P/L from market liquidity/credit premiums.

### Phase 5: Risk Management (ALM Stress Test)
- **Scenario:** Applied the **"Most Likely Scenario"** predicted by the ML model.
- **Analysis:** Conducted **Duration Gap Analysis** to evaluate the sensitivity of Net Asset Value (Equity) under the predicted rising rate environment.

---

## 3. Financial Implications

### "High-Fidelity Risk Quantification"
The backtesting results for H2 2025 demonstrated exceptional accuracy in quantifying market risk.

- **Bear Market Prediction:** The ML model correctly forecasted the **upward trend** in interest rates (Yield Rise), predicting a significant valuation loss for the bond portfolio.
- **P/L Explanation:**
  - **Expected P/L (Model):** -306.99 (10Y), -1,119.71 (50Y)
  - **Actual P/L (Market):** -325.39 (10Y), -1,127.12 (50Y)
  - **Conclusion:** The model's expected P/L explained over **94% of the Actual P/L**. The minimal difference indicates that **Spread Risk was stable**, and the losses were primarily driven by the **Rate Change (Delta)**, which the model successfully predicted.
- **Value:** This proves the system's capability to accurately estimate future capital losses in a rising rate environment, enabling proactive **Duration Shortening** or **Hedging Strategies**.

---

## 4. Dataset Description
*(Raw market data is excluded for compliance.)*
- **Market Data:** Daily KTB Yields (2017.01 ~ 2025.12)
    -  ECOS(BOK, Market Interest rates(Daily))
- **Bond Specs:**
  - 10Y Benchmark: KTB 02625-3506 (Maturity: Jun 2035)
  - 50Y Benchmark: KTB 02750-7409 (Maturity: Sep 2074)
  - KRX Data Marketplace(Bond, Individual Issues Fair Value Valuation Trends)

---

## 5. Directory Structure
```text
ML-Driven-Bond-Attribution-System/
├── yield_curve_model/            # Core Logic & Execution Scripts
│   ├── models.py                 # Classes: NSSModel, ForecastingModel, BondPricing, AttributionModel
│   ├── data_loader.py            # Oracle DB Connection & Data Fetching
│   ├── run_nss_validation.py     # Step 1: NSS Curve Fitting Validation
│   ├── run_forecast_validation.py# Step 2: XGBoost Forecasting Validation
│   ├── run_expected_vs_actual.py # Step 3: P/L Simulation (Expected vs Actual)
│   ├── run_attribution.py        # Step 4: Performance Attribution (Waterfall Chart)
│   └── run_stress_test.py        # Step 5: ALM Stress Testing
│
├── images/                       # Result Visualizations
│   ├── nss_validation.png
│   ├── forecast_validation.png
│   ├── expected_vs_actual_curve.png
│   └── attribution_waterfall.png
│
├── requirements.txt              # Python Dependencies
└── README.md                     # Project Documentation
```

------

# 머신러닝 기반 금리 예측 및 채권 성과 분해 시스템
**(ML-Driven Interest Rate Forecasting and Bond Performance Attribution System)**

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 국고채 금리 예측에 머신러닝(XGBoost) 기법을 도입하고, 이를 전통적인 금융공학 모델(NSS Curve Fitting)과 결합하여 채권 포트폴리오의 미래 손익을 시뮬레이션하는 **End-to-End 리스크 관리 시스템**입니다.

정적인 리스크 측정의 한계를 넘어, 미래의 **수익률 곡선(Yield Curve)** 변화를 예측하고 이에 따른 평가 손익(Valuation P/L)을 정량화함으로써, 급변하는 금리 환경에 선제적으로 대응할 수 있는 **조기 경보 시스템(Early Warning System)** 구축을 목표로 합니다.

## 2. 주요 기능 및 방법론 (Methodology)
본 프로젝트는 총 5단계의 파이프라인으로 구성되어 있습니다.

### Phase 1: 데이터 엔지니어링 & 커브 구성
- **Source:** Oracle DB (2017~2025 국고채 금리).
- **Method:** **Nelson-Siegel-Svensson (NSS)** 모델을 사용하여 만기별 금리 데이터를 연속적인 **수익률 곡선**으로 적합(Calibration).

### Phase 2: ML 예측 (만기별 모델링)
- **Model:** XGBoost Regressor.
- **Approach:** 3년(단기), 10년/50년(장기) 등 만기별 특성을 반영한 **개별 모델(Decoupled Models)**을 구축.
- **Validation:** **Walk-forward Validation**을 통해 2025년 하반기 금리의 **우상향(금리 상승)** 추세를 예측.

### Phase 3: 가치 평가 (Pricing Engine)
- **Engine:** 자체 구축한 `BondPricing` 엔진으로 현금흐름 할인.
- **Logic:** ML 모델이 예측한 **금리 상승분(Difference)**을 반영하여 기말 예상 커브 및 이론가 도출.

### Phase 4: 성과 요인 분해 (Waterfall Analysis)
채권 손익(P/L)을 3가지 핵심 요소(Carry, Rate, Spread)로 분해하여 성과의 원천을 규명.

### Phase 5: 리스크 관리 (ALM Stress Test)
- **Scenario:** ML 예측 시나리오(금리 상승) 기반의 스트레스 테스트 수행.
- **Analysis:** 듀레이션 갭(Duration Gap) 분석을 통해 금리 상승 시 자본(Equity) 감소분 시뮬레이션.

---

## 3. 금융학적 함의 (Financial Implications)

### "정교한 손실 예측 및 리스크 계량화 성공"
백테스팅 결과, 본 시스템은 금리 상승기(Bear Market)의 손실 규모를 매우 높은 정확도로 예측해냈습니다.

- **시장 상황 예측:** ML 모델은 2025년 하반기 금리의 **상승 추세(Yield Rise)**를 정확히 포착하였으며, 이에 따른 채권 평가 손실을 예고했습니다.
- **P/L 설명력:**
  - **예상 손익(Expected P/L):** -306.99 (10Y), -1,119.71 (50Y)
  - **실제 손익(Actual P/L):** -325.39 (10Y), -1,127.12 (50Y)
  - **결론:** 모델의 예상 손익이 실제 손익을 **약 94% 이상 설명**했습니다. 이는 두 손익의 차이인 **Spread Risk(괴리)**가 매우 안정적이었으며, 시장의 손실이 대부분 모델이 예측 가능한 **금리 요인(Delta)**에 의해 주도되었음을 의미합니다.
- **가치:** 이는 본 시스템이 향후 금리 변동 시 발생할 자본 손실을 사전에 정확히 계량화할 수 있으며, 이를 바탕으로 **듀레이션 축소**나 **헤지(Hedge)** 전략을 수립하는 데 신뢰할 수 있는 지표가 됨을 입증합니다.

---

## 4. 데이터셋 설명
*(보안상 원본 데이터 제외)*
- **Market Data:** 2017.01 ~ 2025.12 국고채 일별 금리
  -  ECOS(한국은행, 시장금리(일별))
- **Bond Specs:**
  - 10년 지표물: 국고02625-3506 (만기: 2035-06)
  - 50년 지표물: 국고02750-7409 (만기: 2074-09)
  - KRX Data Marketplace(채권, 개별종목 시가평가 추이)

---

## 5. 디렉토리 구조 (Directory Structure)
```text
ML-Driven-Bond-Attribution-System/
├── yield_curve_model/            # 핵심 로직 및 실행 스크립트
│   ├── models.py                 # 모델 클래스 (NSS, 예측, 프라이싱, 성과분해)
│   ├── data_loader.py            # DB 연결 및 데이터 로드
│   ├── run_nss_validation.py     # [1단계] NSS 커브 적합도 검증
│   ├── run_forecast_validation.py# [2단계] XGBoost 금리 예측 검증
│   ├── run_expected_vs_actual.py # [3단계] 예상 손익 vs 실제 손익 시뮬레이션
│   ├── run_attribution.py        # [4단계] 성과 요인 분해 (Waterfall 차트 생성)
│   └── run_stress_test.py        # [5단계] ALM 스트레스 테스트 수행
│
├── images/                       # 결과 시각화 자료
│   ├── nss_validation.png
│   ├── forecast_validation.png
│   ├── expected_vs_actual_curve.png
│   └── attribution_waterfall.png
│
├── requirements.txt              # 파이썬 의존성 패키지
└── README.md                     # 프로젝트 설명서
