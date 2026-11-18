# 레시피 추천 시스템

Food.com 레시피의 이름과 재료, 사용자 평점 이력으로 개인화된 Top-10을 추천합니다.
현재는 콘텐츠 기반 필터링 구현을 포함하며, 팀원의 모델은 추후 같은 저장소에 통합할 예정입니다.

## 파일

| 파일 | 역할 |
|---|---|
| content_based_filtering.py | TF-IDF 모델, 설정 비교, 사용자별 평가 및 추천 |
| requirements.txt | 실험에 사용한 Python 라이브러리 버전 |
| .gitignore | 데이터, 실행 중간 파일 및 작업 자료 제외 |
| 협업필터링.ipynb | SVD 기반 행렬 분해 협업 필터링 모델 및 추가 튜닝 실험 |
| 협업 필터링 보고서.docx | SVD 협업 필터링 실험 및 분석 보고서 |

## 모델

재료 40%와 이름 60%의 TF-IDF 특징을 결합합니다. 레시피 텍스트에 SVD 32차원을 학습한 뒤 첫 8개 공통 성분을 완전히 감쇠하고, 보정된 특징 공간에서 cosine 유사도를 계산합니다.
사용자 프로필은 Train에서 평점 4점 이상을 준 레시피의 동일 가중 평균으로 구성합니다. SVD의 학습 대상은 사용자 평점 행렬이 아닌 레시피 텍스트입니다.

## 실행

실험 환경은 Python 3.8.10이며 라이브러리 버전은 requirements.txt에 기록했습니다.

    python -m pip install -r requirements.txt

팀에서 전처리한 CSV 4개를 전처리_완료/에 준비합니다. 원본 데이터는 저장소에 포함하지 않습니다.

    전처리_완료/
      recipe_features.csv
      train_ratings.csv
      validation_ratings.csv
      test_ratings.csv

평점 파일에는 user_id, recipe_id, rating, date, 레시피 파일에는 recipe_id, name, ingredients_text, tags_text, content_text, minutes 컬럼을 사용합니다.

    python content_based_filtering.py --data-dir 전처리_완료

기본 실행은 공통 성분 보정 4개 설정을 전체 Validation에서 비교하고, 선택한 모델을 Test에서 평가합니다. 결과는 outputs/residual_content/에 저장합니다.

| 옵션 | 실행 내용 |
|---|---|
| --legacy | 발표 당시 두 설정 비교, Recall 우선 선택 |
| --revision | TF-IDF·프로필 8개 설정 비교 |
| --advanced | LSA 콘텐츠 표현 7개 설정 비교 |
| --residual | 공통 성분 보정 4개 설정 비교, 기본 실행과 동일 |
| --example-only | 전체 평가 없이 사용자 187412의 추천 출력 |
| --user-id ID | 예시 사용자 변경 |
| --output-dir 경로 | 결과 저장 위치 변경 |

    python content_based_filtering.py --data-dir 전처리_완료 --example-only

Colab에서는 코드와 CSV를 /content에 올리고 실행합니다.

    %run /content/content_based_filtering.py --data-dir /content

## 평가 조건과 결과

- Train과 평가 데이터에 각각 평점 4점 이상 기록이 있는 사용자만 평가합니다.
- 추천 후보는 Train 전체 레시피 226,590개이며, 사용자의 Train 평가 항목은 평점과 관계없이 모두 제외합니다.
- 평가 데이터의 평점 4점 이상 레시피가 정답입니다.
- 사용자별 Precision@10, Recall@10, F1@10을 계산한 뒤 평균냅니다. 무적중 사용자도 포함합니다.
- Validation 20,774명에서 Macro F1을 우선 기준으로 설정을 선택하고, Test 1,321명에서 평가했습니다.

| Test 모델 | Precision@10 | Recall@10 | Macro F1@10 | 적중수 |
|---|---:|---:|---:|---:|
| 발표 당시 TF-IDF | 0.00030280 | 0.00227101 | 0.00052761 | 4 |
| 공통 성분 보정 모델 | 0.00045420 | 0.00177860 | 0.00059726 | 6 |
| SVD 협업 필터링 | 0.0002 | 0.0008 | 0.0001 | - |

같은 Test 분할에서 F1은 13.20%, Precision은 50% 상승했고 Recall은 21.68% 하락했습니다. 적중수가 적고 사용자별 F1 차이의 95% 부트스트랩 구간이 0을 포함하므로 일반적인 성능 우위를 확정할 수는 없습니다. Test를 이전 개발 단계에서도 확인한 후속 실험이라는 한계가 있습니다.

## 데이터와 참고 자료

- [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
- [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

팀이 제공한 Train·Validation·Test 분할을 사용하며, 원본 Kaggle 파일만으로 동일 분할을 만드는 전처리 코드는 현재 포함되어 있지 않습니다. 라이브러리의 역할과 주요 연산은 Python 코드의 한국어 주석에 설명했습니다.
