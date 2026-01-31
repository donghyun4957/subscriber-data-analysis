# 머신러닝 기반 신문 구독자 이탈 분석

> SK네트웍스 Family AI 캠프

---

## Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Data Selection & Structure](#data-selection--structure)
4. [Data Preprocessing](#data-preprocessing)
5. [EDA](#eda)
6. [Machine Learning](#machine-learning)
7. [Limitations & Insights](#limitations--insights)
8. [Demo Page](#demo-page)
9. [Team Retrospective](#team-retrospective)

---

## Project Overview

### 프로젝트 소개

본 프로젝트는 **전통 언론사가 직면한 구독자 이탈 문제**를 데이터 기반으로 해결하는 것을 목표로 합니다.

온라인 및 오프라인 신문 구독 데이터를 분석하여 **구독자의 이탈 가능성**을 예측하고, 그 결과를 바탕으로 **맞춤형 유지 전략**을 수립함으로써, 신문사의 **지속 가능한 경영**과 **경쟁력 강화**에 기여하고자 합니다.

### 프로젝트 필요성

<p align="center">
  <img src="./images/newss.png" width="600">
</p>

**신문 구독 감소 추세**
- 디지털 미디어 확산으로 정기 구독 인구 지속 감소
- 스마트폰·인터넷 보급, 뉴스 플랫폼·유튜브 등 영상 기반 채널 성장
- 실시간·무료·맞춤형 콘텐츠 접근성 증가로 정보 소비 패턴 변화

**온라인 뉴스와 전통 신문의 차이**
- 신문: 인쇄·편집 등 다단계 검증 과정을 거침
- 온라인 뉴스: 클릭 수·조회 수 중심, 자극적인 제목·단편적 내용 빈번
- 전통 언론의 사실·맥락 전달 역할은 여전히 필요

**신문사의 현황과 과제**
- 신문사는 뉴스 생산 기업이며, 구독료와 광고 수익이 주요 재원
- 전통 구독 기반 수익 모델 붕괴 → 재정 압박 심화
- 구독자 이탈 방지 전략은 신문사의 생존과 지속 가능성에 필수

### 프로젝트 목표

- 데이터 기반으로 이탈 위험 구독자를 조기에 식별하여 맞춤형 할인, 콘텐츠 추천 등 유지 전략을 실행
- 장기적으로 안정적인 구독 기반 확보를 통해 변화하는 미디어 환경 속에서 경쟁력 유지
- 건강한 정보 전달 체계 유지로 사회적 기여

---

## Technology Stack

### WBS
![WBS](images/wbs.png)

### 기술 스택

| 분류 | 기술/도구 |
|------|----------|
| 언어 | Python |
| 라이브러리 | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Streamlit |
| 협업 툴 | GitHub, Git |

---

## Data Selection & Structure

### 데이터
- 캘리포니아 신문 구독자 데이터
- 출처: https://www.kaggle.com/datasets/andieminogue/newspaper-churn

### 데이터 구조

**분석 타겟 컬럼**
- `Subscriber` : 구독자 이탈 여부 (분류 대상)

**주요 변수**

| 변수명 | 설명 |
|--------|------|
| SubscriptionID | 구독자 고유 ID |
| HH Income | 가구 수입 |
| Home Ownership | 주거 형태(자가/임대 여부) |
| Ethnicity | 민족 |
| dummy for Children | 자녀 유무 더미 변수 |
| Year Of Residence | 현재 거주 기간(연 단위) |
| Age range | 나이 구간 |
| Language | 사용 언어 |
| Address | 주소 |
| State | 주(state) |
| City | 도시 |
| County | 행정구역 |
| Zip Code | 우편번호 |
| weekly fee | 주당 구독료 |
| Deliveryperiod | 배달 요일 또는 배달 주기 |
| Nielsen Prizm | 인구통계 세분화 모델 코드 |
| reward program | 보상 프로그램 수령 횟수 |
| Source Channel | 유입 경로 |

---

## Data Preprocessing

결측치 및 이상치 처리, 파생변수, 변수명 정제 등 전처리 수행.

### 수치형, 범주형 데이터 조회

**연속형**
```
['SubscriptionID', 'Year Of Residence', 'Zip Code', 'reward program']
```

**범주형**
```
['HH Income', 'Home Ownership', 'Ethnicity', 'dummy for Children', 'Age range', 'Language', 'Address', 'State', 'City', 'County', 'weekly fee', 'Deliveryperiod', 'Nielsen Prizm', 'Source Channel', 'Subscriber']
```

### 파생변수 생성
- 이탈 여부를 0/1로 변환하여 데이터들 간의 이탈률 여부 분석을 용이하게 하기 위해 `Subscriber` 변수로부터 `is_churned` 변수 생성

### 결측치 처리

**결측치 조회**
```
SubscriptionID           0
HH Income                0
Home Ownership           0
Ethnicity                0
dummy for Children       0
Year Of Residence        0
Age range              108
Language              1007
Address                  0
State                    0
City                     0
County                   0
Zip Code                 0
weekly fee             186
Deliveryperiod           0
Nielsen Prizm          129
reward program           0
Source Channel           0
Subscriber               0
```

**결측치 처리 방법**
- `Age`, `weekly fee`, `Nielsen Prizm`: 결측치 제거
- `Language`: 결측치가 약 6.3%로 상대적으로 큰 수치이나, 'unknown'으로 처리 시 이탈률에 유의미한 차이가 없어 'unknown'으로 대체
- `Ethnicity`의 'unknown' 범주: Language가 결측인 데이터들은 항상 Ethnicity 범주가 'unknown'임을 발견하여 유지 결정

### 이상치 조회
- `Year Of Residence`: 큰 이상치는 존재하지 않으므로 처리하지 않음
- `reward program`: 극단치가 존재하지만, 할인 관련 혜택을 받는 횟수를 나타내는 특성이므로 유지

### 중복값 제거
- `Delivery Period`에 중복된 범주가 있음 (7Day, 7day 등)
- 범주명을 소문자로 변환하여 중복 제거

### 최종 데이터
- 이탈률과 무관하거나 다른 변수와 중복 특성을 가진 데이터 제거 (`SubscriptionID`, `State`, `Zip Code`, `Address`)
- 파생변수 생성에 사용된 변수 제거 (`Subscriber`)

```
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   HH Income           15438 non-null  object
 1   Home Ownership      15438 non-null  object
 2   Ethnicity           15438 non-null  object
 3   dummy for Children  15438 non-null  object
 4   Year Of Residence   15438 non-null  int64
 5   Age range           15438 non-null  object
 6   Language            15438 non-null  object
 7   City                15438 non-null  object
 8   County              15438 non-null  object
 9   weekly fee          15438 non-null  object
 10  Deliveryperiod      15438 non-null  object
 11  Nielsen Prizm       15438 non-null  object
 12  reward program      15438 non-null  int64
 13  Source Channel      15438 non-null  object
 14  is_churned          15438 non-null  int64
```

---

## EDA

### 이탈/잔류 전체 비율
![Churn Rate](images/1.churn_rate.png)

- 이탈 고객이 전체의 약 80%로, 데이터가 심하게 불균형되어 있음
- 데이터 분석 시 전체 이탈률 대비 범주별 이탈률의 차이를 시각화
- 범주형 변수와 이탈여부 간의 관련성을 확인하기 위해 카이제곱 검정, 크래머 V계수 활용

**카이제곱검정**: 범주형 데이터의 관측 빈도와 기대 빈도 사이의 차이를 계산하여 검정하는 통계적 방법

$$
\chi^2 = \sum_{i=1}^{r} \sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

**크래머 V계수**: 카이제곱 독립성 검정의 효과의 크기 측정

$$
V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, \, c-1)}}
$$

### 변수별 이탈률 분석

#### 집 소유 여부별 이탈률
![Home Ownership](images/3.home.png)
```
카이제곱 P-value: 2.3432e-51
Cramér's V: 0.1213
```
→ 집을 렌탈하는 사람들(Renter)이 상대적으로 이탈률이 높음

#### 민족 & 언어별 이탈률
![Ethnicity](images/4-1.ethnicity.png)
```
카이제곱 P-value: 1.1848e-31
Cramér's V: 0.1164
```

![Language](images/4-2.language.png)
```
카이제곱 P-value: 6.6138e-27
Cramér's V: 0.0953
```
→ 민족 및 언어별로 이탈률 차이가 있으나 지배적인 영향은 아님

#### 나이 구간별 이탈률
![Age](images/5.age.png)
```
카이제곱 P-value: 1.3018e-120
Cramér's V: 0.1964
```
→ 나이가 증가함에 따라 이탈률이 계속 감소하는 추세

#### 소득 구간별 이탈률
![Income](images/6.income.png)
```
카이제곱 P-value: 4.9239e-39
Cramér's V: 0.1210
```
→ 소득이 증가함에 따라 이탈률이 계속 감소하는 추세

#### 유입 경로별 이탈률
![Source Channel](images/7.source_channel.png)
```
카이제곱 P-value: 2.6269e-233
Cramér's V: 0.2710
```
→ 유입 경로별 상대적으로 이탈률의 차이가 큼

#### 배달 주기별 이탈률
![Delivery Period](images/8.delivery_periond.png)
```
카이제곱 P-value: 1.3454e-98
Cramér's V: 0.1771
```
→ 주당 배달 횟수와 요일에 따라 이탈률 차이가 있음

#### 자녀 유무별 이탈률
![Dummy](images/9.dummy.png)
```
카이제곱 P-value: 6.4890e-01
Cramér's V: 0.0037
```
→ 자녀 유무에 따른 전체 이탈률 차이는 거의 없음

**기타 변수들과의 상관관계 분석**

Age와의 관계:
![Age Correlation](images/9-1.age.png)
```
카이제곱 p-value = 1.8784e-116
```
→ 자녀 유무에 따라 연령별 이탈률 차이가 있고, 전반적으로 연령대가 높아질수록 이탈률이 낮아지는 경향

Income과의 관계:
![Income Correlation](images/9-2.income.png)
```
카이제곱 p-value = 2.5852e-36
```
→ 소득 구간별로 자녀 유무에 따른 이탈률 차이가 있음

#### 도시 및 행정구역별 이탈률
![City](images/10.city.png)
```
카이제곱 P-value: 9.5371e-15
Cramér's V: 0.0964
```

![County](images/10-2.county.png)
```
카이제곱 P-value: 1.7125e-04
Cramér's V: 0.0360
```

#### 주당 구독료별 이탈률
![Weekly Fee](images/11.weekly_fee.png)
```
카이제곱 P-value: 0.0000e+00
Cramér's V: 0.4133
```
→ 구독료가 증가함에 따라 이탈률 차이가 큼

#### 인구통계 세분화 모델 코드별 이탈률
![Nielsen](images/12.nielsen.png)
```
카이제곱 P-value: 3.4017e-48
Cramér's V: 0.1257
```

#### 거주 기간별 이탈률
![Year of Residence](images/13.year_of_residence.png)
```
카이제곱 P-value: 2.2910e-140
Cramér's V: 0.2097
```
→ 거주기간이 길수록 이탈률이 감소하는 추세

#### 보상 프로그램 수령 횟수별 이탈률
![Reward](images/14.reward.png)
```
카이제곱 P-value: 1.0287e-52
Cramér's V: 0.1311
```
→ 할인을 많이 받은 사람들이 이탈률이 적음

### EDA 결론
- 카이제곱 검정 결과, `dummy for Children` 변수를 제외한 모든 변수가 이탈과 유의미한 관련성을 보임
- 기타 변수들과 `dummy for Children`을 함께 분석한 결과, 유의미한 이탈률과의 관련성 확인
- 모든 변수들이 이탈률과 관련이 있음을 확인하였으므로, 모든 변수를 모델의 Feature로 사용하기로 결정

---

## Machine Learning

### 학습을 위한 전처리

#### 나이 및 소득 수치형 특성으로 변환

원본 데이터의 나이와 소득은 구간별 범주형 데이터로 제공됨. 대부분의 머신러닝 알고리즘은 입력을 수치형으로 받으므로 변환 필요.

**나이 변환(Age)**
- 통계학적으로 구간 데이터는 계급 중앙값으로 추정하는 것이 표준
- 본 프로젝트에서는 절단 정규분포(truncated normal)에서 난수를 추출하는 확률적 대치(stochastic imputation) 사용
- 분산 축소 방지 및 변동성 유지에 유리

**소득 변환(HH Income)**
- 소득 분포는 일반적으로 로그 정규분포
- 각 구간의 하한·상한 범위 내에서 로그 정규분포 기반 난수 추출

#### 데이터 불균형 해결

원본 데이터에서 `is_churned=1`이 압도적으로 많음(약 20:80 비율). 오버샘플링 기법 SMOTENC 도입.

```python
# 타깃 / 피처 분리
y = converted_df['is_churned'].astype(int)
X = converted_df.drop(columns=['is_churned']).copy()

# 범주형 / 수치형 컬럼 지정
cat_cols = [
    'Home Ownership','Ethnicity','dummy for Children',
    'Language','City','County','weekly fee',
    'Deliveryperiod','Nielsen Prizm','Source Channel'
]
num_cols = ['Year Of Residence', 'reward program', 'Age', 'Income']

# SMOTENC 적용
smote = SMOTENC(
    categorical_features=cat_idx,
    random_state=42,
    k_neighbors=k_neighbors
)
X_res, y_res = smote.fit_resample(X_enc, y)
```

**SMOTENC 적용 결과**

| 구분 | 클래스 1 개수 | 클래스 0 개수 |
|------|--------------|--------------|
| 변경 전 | 12,434 | 3,004 |
| 변경 후 | 12,434 | 12,434 |

### 모델 선정 및 사전 학습

각 조원은 하나의 모델을 맡아 총 5개의 모델을 구성. `GridSearchCV`를 활용해 다양한 하이퍼파라미터 조합을 탐색.

**1) XGBoost**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.87 | 0.89 | 0.88 | 2,487 |
| 1 | 0.89 | 0.86 | 0.87 | 2,487 |
| Accuracy | | | 0.88 | 4,974 |

**2) Random Forest**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.87 | 0.87 | 0.87 | 3,109 |
| 1 | 0.87 | 0.87 | 0.87 | 3,108 |
| Accuracy | | | 0.87 | 6,217 |

**3) Decision Tree**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.79 | 0.84 | 0.82 | 2,487 |
| 1 | 0.83 | 0.78 | 0.81 | 2,487 |
| Accuracy | | | 0.81 | 4,974 |

**4) SVM**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.565 | 0.527 | 0.545 | 3,143 |
| 1 | 0.547 | 0.585 | 0.566 | 3,074 |
| Accuracy | | | 0.556 | 6,217 |

**5) Gradient Boosting**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.89 | 0.89 | 0.89 | 2,487 |
| 1 | 0.89 | 0.89 | 0.89 | 2,487 |
| Accuracy | | | 0.89 | 4,974 |

**결론:** Gradient Boosting이 가장 높은 정확도와 균형 잡힌 성능을 보여 최종 모델로 선정.

### 최종 모델 성능 고도화

최종 모델로 선정된 Gradient Boosting의 최적 하이퍼파라미터:

| Hyperparameter | Value |
|----------------|-------|
| learning_rate | 0.225 |
| max_depth | 8 |
| n_estimators | 400 |
| subsample | 0.95 |

교차 검증 결과:
```
Best model CV mean F1: 0.8762 (+/- 0.0032)
```

임계치(Threshold)를 0.32로 조정하여 재현율(Recall) 향상:

**Best Threshold:** 0.32
**Macro F1:** 0.8900

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.90 | 0.88 | 0.89 | 2,487 |
| 1 | 0.88 | 0.90 | 0.89 | 2,487 |
| Accuracy | | | 0.89 | 4,974 |

![ROC AUC](images/best_model_ROC_AUC.png)

![Precision Recall Curve](images/best_model_precision_recall_curve.png)

![Threshold Line](images/best_model_thresholdline.png)

### SMOTENC 효과 검증

원본 데이터(오버샘플링 미적용)를 동일한 모델에 학습시켜 비교:

**SMOTENC 미적용 결과:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 | 0.62 | 0.44 | 0.51 | 601 |
| 1 | 0.87 | 0.94 | 0.90 | 2,487 |
| Accuracy | | | 0.84 | 3,088 |

**Best Threshold:** 0.62
**Macro F1:** 0.7090

**결론:** SMOTENC 적용 시 소수 클래스 예측력이 크게 개선되었으며, 전체적인 Macro F1 점수도 향상됨.

### 최적 모델 성능에 대한 고찰

<p align="center">
  <img src="images/best_model_calibration_curve.png" width="600">
</p>

- X축 (Mean predicted probability): 예측 확률을 비슷한 구간별로 묶어 평균한 값
- Y축 (Fraction of positives): 해당 구간에서 실제로 이탈한 비율
- 0.0~0.3: 실제 이탈 비율이 예측보다 높아 위험 과소평가 경향
- 0.4~0.7: 예측과 실제가 거의 일치
- 0.8 이상: 실제로도 대부분 이탈, 약간의 오차 존재

<p align="center">
  <img src="images/best_model_feature_importances.png" width="600">
</p>

- 모델이 부여한 Feature importance 경향성과 크래머 V 분석의 상관 계수 경향성이 비슷하여 EDA 과정에서 특성 선별이 유의미함을 확인
- Feature importance와 크래머 V 분석의 경향성이 완전히 같지 않다는 것은 EDA만으로 포착할 수 없는 다변량 변수들 간의 관계들을 머신러닝 모델이 분석할 수 있다는 의의가 있음

---

## Limitations & Insights

### 한계점

**Oversampling 방식의 한계**

SMOTENC는 학습 데이터에서 관측된 국소적 분포를 기반으로 데이터를 생성하기 때문에 실제 모집단 분포를 보장하지 못함. 클래스 경계 부근의 노이즈나 범주형 변수 조합의 비현실성으로 인해 모델의 일반화 성능이 저하될 가능성이 있음.

**시간적 제약**

다양한 하이퍼파라미터 조합을 탐색함으로써 모델의 성능을 더욱 개선할 수 있음을 확인했으나, 시간적 제약으로 인해 충분한 조합을 실험하지 못함. 특히 SVM의 경우 약 900분이 소요되어 테스트 범위가 제한됨.

### 인사이트

현재 모델은 캘리포니아 지역 구독자 데이터를 기반으로 학습되었지만, 향후 한국 신문사 구독자 데이터를 확보·재학습함으로써 한국 시장 특성에 최적화된 예측 성능을 확보할 수 있습니다.

이 모델을 CRM 시스템에 통합하면:
- 구독자 유지율 향상
- 광고·구독 기반 수익의 장기적 안정화
- 데이터 기반의 경영 의사결정 지원

등의 실질적인 비즈니스 성과를 창출할 수 있습니다.

---

## Demo Page

![Demo](images/Animation.gif)

---

## References
- Aitchison, J., & Brown, J. A. C. (1957). The lognormal distribution. Cambridge University Press.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics.
- van Buuren, S. (2018). Flexible imputation of missing data (2nd ed.). CRC Press.
