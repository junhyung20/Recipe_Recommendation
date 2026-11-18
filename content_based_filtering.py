"""콘텐츠 기반 레시피 추천 및 사용자별 Macro P/R/F1@10.

실행: python content_based_filtering.py --data-dir 전처리_완료
Colab: %run /content/content_based_filtering.py --data-dir /content
TF-IDF 8설정 비교는 --revision, LSA 비교는 --advanced, 공통 성분 보정 비교는 --residual.
공통 사용자 예시만 확인하려면 --example-only (전체 Validation/Test 반복 없음).
F1@10은 Top-10과 관측된 긍정 레시피 집합의 조화평균이다. 모든 미관측 항목을
진짜 부정으로 확정하는 일반 분류 F1과 구분한다. 미관측 선호는 평가할 수 없다.
과거 Test 결과는 이미 알려졌으므로 이번 실험도 완전히 새로운 미공개 holdout이 아니다.

공통 Python 요소: dict/list/set은 설정/순서/중복 없는 정답 집합, tuple은 고정 범위,
len/range/enumerate/zip은 크기/배치/순번/병렬 순회, int/float/str/bool은 자료형 변환,
next는 첫 생성 결과, yield는 배치 결과를 한 사용자씩 전달, raise는 조건 위반 중단이다.
if/for와 리스트 내포는 조건/반복 처리, **는 설정 사전 전개, @는 행렬 곱이다.
"""
# Path: 운영체제에 맞는 경로 결합, is_file/is_dir 검사, mkdir 폴더 생성,
# read_text/write_text는 UTF-8 입출력, resolve/parent는 절대 경로/상위 폴더이다.
from pathlib import Path
# MethodType: 공통 클래스에 실험별 점수 함수를 연결한다.
from types import MethodType
# argparse: --data-dir 등의 명령행 인수를 자료형에 맞게 해석한다.
import argparse
# copy.copy: 이력 희소행렬을 복제하지 않고 설정별 모델 껍데기를 만든다.
import copy
# hashlib.sha256: 결과 재사용이 아닌 입력/행렬 동일성 확인용 해시이다.
import hashlib
# json.dumps: 설정/진단을 사람이 읽을 수 있는 JSON으로 직렬화한다.
import json
# time.perf_counter: 모델별 실제 계산 시간을 초 단위로 측정한다.
import time
# NumPy: float32(메모리 절약), int64(ID), 배열/내적 보조/정렬/집합/평균을 처리한다.
# sort/lexsort는 정렬, partition/argpartition은 전체 정렬 없이 Top-k 경계를 찾는다.
# isfinite/flatnonzero/isin/intersect1d는 유효값/위치/포함/사용자 교집합을 구한다.
# ascontiguousarray는 연속 메모리, sqrt는 특징 가중치의 제곱근이다.
import numpy as np
# pandas: read_csv/DataFrame으로 표를 읽고 생성한다. loc/isin/ge/eq는 행 필터,
# groupby/apply(set)는 사용자별 정답 집합, merge는 이름 결합, mean/sum/median은 통계,
# to_numpy/get_indexer는 ID→행 위치, to_csv/to_string은 결과 저장/실제 출력이다.
# sort_values/reset_index/set_index/reindex는 행 정렬·인덱스 정렬을 수행한다.
# duplicated/isna/any/all은 데이터 검증, MultiIndex.intersection은 평가쌍 중복 검사이다.
import pandas as pd
# CSR: 0이 아닌 값만 저장. data/indices/indptr는 값/열/행 경계, hstack은 특징 열 결합.
from scipy import sparse
# TfidfVectorizer: 텍스트→어휘별 TF-IDF 희소행렬. fit_transform은 어휘/IDF 학습과 변환.
from sklearn.feature_extraction.text import TfidfVectorizer
# normalize: 각 행 L2 길이를 1로 만들어 내적이 cosine 유사도가 되게 한다.
from sklearn.preprocessing import normalize
# TruncatedSVD는 평점 행렬이 아닌 레시피 TF-IDF를 압축하는 LSA 콘텐츠 표현이다.
# random_state는 재현성, n_iter는 근사 분해 정확도, n_components는 잠재 텍스트 차원.
# API 참고: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
from sklearn.decomposition import TruncatedSVD
# pyplot: subplots로 실제 수치 그래프 생성, savefig로 PNG 저장, show/close로 표시/해제.
import matplotlib.pyplot as plt


VALIDATION_CONFIGS = [
    ('ingredients40_names60', {'ingredients': 0.4, 'name': 0.6}),
    ('ingredients50_names30_tags20', {'ingredients': 0.5, 'name': 0.3, 'tags': 0.2}),
]
FINAL_MODEL_NAME = 'ingredients40_names60'

def save_json(path, value):
    """path에 value를 UTF-8 JSON 저장. ensure_ascii=False는 한국어 유지, indent=2는 가독성."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def topk(scores, ids, k):
    """scores에서 k개 선택. ids는 점수 동률일 때 작은 레시피 ID를 먼저 고르는 기준이다."""

    k = min(k, int(np.isfinite(scores).sum()))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)
    needed = k - len(above)
    if len(tied) > needed:
        tied = tied[np.argpartition(ids[tied], needed - 1)[:needed]]
    chosen = np.concatenate([above, tied])
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]

class ContentModel:
    """Train 이력과 전체 후보를 공유하며 프로필·추천·매크로 평가를 제공하는 기본 클래스."""
    def __init__(self, recipes, train, max_features, min_df, ngram_range, profile_weight):
        """recipes/train: 원본 표. 기존 API의 TF 인수는 호환용이며 실제 TF는 build_fields 담당.
        profile_weight는 uniform(동일 선호), rating(평점-3) 중 선택한다.
        """
        self.recipes = recipes[recipes.recipe_id.isin(train.recipe_id)].sort_values('recipe_id').reset_index(drop=True)
        self.ids = self.recipes.recipe_id.to_numpy()
        self.item_index = pd.Index(self.ids)
        self.users = pd.Index(np.sort(train.user_id.unique()))
        # 기존의 content_text 임시 TF-IDF는 즉시 폐기되므로 재학습 낭비를 제거했다.
        self.X = None
        rows = self.users.get_indexer(train.user_id)
        cols = self.item_index.get_indexer(train.recipe_id)
        shape = (len(self.users), len(self.ids))
        self.seen = sparse.csr_matrix((np.ones(len(train), dtype=np.int8), (rows, cols)), shape=shape)
        positive = train.rating.to_numpy() >= 4
        self.positive_rows, self.positive_cols = rows[positive], cols[positive]
        self.positive_ratings = train.rating.to_numpy(dtype=np.float32)[positive]
        self.positive_users = np.sort(train.loc[positive, 'user_id'].unique())
        self.set_weight(profile_weight)

    def set_weight(self, weight):
        """양성(4/5점) 이력만 W에 저장. rating은 1/2 가중치로 5점 선호를 강조한다."""

        values = self.positive_ratings - 3 if weight == 'rating' else np.ones_like(self.positive_ratings)
        self.W = sparse.csr_matrix((values, (self.positive_rows, self.positive_cols)), shape=self.seen.shape)

    def eligible(self, evaluation):
        """Train 및 evaluation 각각 4점 이상 이력이 있는 사용자 교집합을 반환한다."""
        return np.intersect1d(self.positive_users, evaluation.loc[evaluation.rating.ge(4), 'user_id'].unique())

    def recommend(self, users, k=10, batch_size=32):
        """users에게 k개 추천. batch_size=32는 후보 전체를 유지하면서 점수 메모리 제한."""

        if not np.isin(users, self.positive_users).all():
            raise ValueError('TRAIN 양성 이력이 없는 사용자')
        for start in range(0, len(users), batch_size):
            block = users[start:start + batch_size]
            user_rows = self.users.get_indexer(block)
            profiles = normalize(self.W[user_rows] @ self.X)
            scores = (self.X @ profiles.toarray().T).T
            for j, (uid, row) in enumerate(zip(block, user_rows)):
                seen = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row + 1]]
                scores[j, seen] = -np.inf
                chosen = topk(scores[j], self.ids, k)
                yield int(uid), self.ids[chosen], scores[j, chosen]

    def evaluate(self, evaluation, k=10, batch_size=32):
        """evaluation은 정답 계산에만 사용. 사용자별 P=hits/k, R=hits/정답수, F1의 평균."""
        users = self.eligible(evaluation)
        if not len(users):
            raise ValueError('평가 대상 사용자가 없음')
        positive = evaluation[evaluation.rating.ge(4) & evaluation.user_id.isin(users)]
        truth = positive.groupby('user_id').recipe_id.apply(set).to_dict()
        rows = []
        for uid, predictions, _ in self.recommend(users, k, batch_size):
            hits = len(set(predictions) & truth[uid])
            precision = hits / k
            recall = hits / len(truth[uid])
            # 사용자별 조화평균을 먼저 구한다. 전체 P/R 평균의 조화평균과 다르다.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(dict(user_id=uid, recommended=len(predictions), hits=hits,
                relevant=len(truth[uid]), precision_at_10=precision, recall_at_10=recall, f1_at_10=f1))
            if len(rows) % 2048 == 0:
                print(f'평가 진행: {len(rows):,}/{len(users):,}명', flush=True)
        details = pd.DataFrame(rows)
        if not details.recommended.eq(k).all():
            raise AssertionError('요청한 개수보다 적게 추천된 사용자가 있습니다.')
        metrics = dict(users=len(users), precision_at_10=float(details.precision_at_10.mean()),
            recall_at_10=float(details.recall_at_10.mean()), f1_at_10=float(details.f1_at_10.mean()), hits=int(details.hits.sum()),
            ground_truth_positives=int(details.relevant.sum()))
        return metrics, details

def build_fields(recipes, needed):
    """needed 특징을 별도 TF-IDF로 학습. recipes는 Train에 등장하는 전체 레시피다.

    min_df=2는 일회성 잡음 제거, max_df는 너무 일반적인 단어 제거 비율,
    max_features는 어휘 수 상한, ngram_range=(1,2)는 단어/인접 두 단어 반영.
    sublinear_tf=True는 TF를 1+log(TF)로 완화하고 기본 smooth_idf=True로 IDF 평활화.
    token_pattern은 한 글자/밑줄 포함 단어 유지, lowercase=True와 norm='l2'는 기본값.
    fillna('')는 결측 텍스트를 빈 문서로 처리한다. 설정은 Baseline 실험값을 유지한다.
    """
    matrices = {}
    specs = {
        'ingredients': dict(column='ingredients_text', min_df=2, max_df=.95, max_features=20000, ngram_range=(1,1)),
        'name': dict(column='name', min_df=2, max_df=.8, max_features=60000, ngram_range=(1,2)),
        'tags': dict(column='tags_text', min_df=2, max_df=.4, max_features=10000, ngram_range=(1,1)),
    }
    for field in needed:
        spec = specs[field].copy()
        column = spec.pop('column')
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', dtype=np.float32,
                                     sublinear_tf=True, **spec)
        matrices[field] = vectorizer.fit_transform(recipes[column].fillna('')).tocsr()
        print('FIELD', field, matrices[field].shape, matrices[field].nnz, flush=True)
    return matrices

def fast_recommend(self, users, k=10, batch_size=32):
    """W@X의 양성 평균 방향을 L2 정규화하고 전체 후보 X와 내적한다.
    seen의 모든 Train 평가(낮은 평점 포함)는 -inf로 마스킹한 뒤 결정적으로 Top-k 추출.
    """
    if not np.isin(users, self.positive_users).all():
        raise ValueError('Train 양성 이력이 없는 사용자')
    for start in range(0, len(users), batch_size):
        block = users[start:start+batch_size]
        rows = self.users.get_indexer(block)
        profiles = normalize(self.W[rows] @ self.X)
        scores = np.ascontiguousarray((self.X @ profiles.toarray().T).T)
        for j, (uid, row) in enumerate(zip(block, rows)):
            seen = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row+1]]
            scores[j, seen] = -np.inf
            selected = topk(scores[j], self.ids, k)
            yield int(uid), self.ids[selected], scores[j, selected]

def export_team_example(model, recipes, train, test, output_dir, user_id=187412,
                        full_test_evaluation_performed=False):
    """공통 user_id의 Train 선호/실제 추천/양성 Test 일치 표시. 정답은 추천 후 대조만 한다.
    output_dir=None이면 저장 생략, full_test_evaluation_performed는 실행 범위 설명용이다.
    """
    output_dir=Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True,exist_ok=True)
    if test[['user_id','recipe_id','rating']].isna().any().any() or test.duplicated(['user_id','recipe_id']).any():
        raise ValueError('Test: 결측 또는 사용자–레시피 중복')
    if not test.rating.isin([1,2,3,4,5]).all():
        raise ValueError('Test 평점 범위 오류')
    if not test.recipe_id.isin(recipes.recipe_id).all():
        raise ValueError('Test 레시피 특성 누락')
    if len(pd.MultiIndex.from_frame(train[['user_id','recipe_id']]).intersection(pd.MultiIndex.from_frame(test[['user_id','recipe_id']]))):
        raise ValueError('Train/Test 평가쌍 중복')
    test_users=model.eligible(test)
    if output_dir is not None:
        pd.DataFrame({'user_id':test_users}).to_csv(output_dir/'eligible_users_test.csv',index=False)
        pd.DataFrame({'recipe_id':model.ids}).to_csv(output_dir/'candidate_recipe_ids.csv',index=False)
    print('Test 평가 조건을 만족하는 사용자:',len(test_users))
    if user_id not in test_users:
        raise ValueError(f'사용자 {user_id}는 Train/Test 양성 사용자 교집합에 없습니다.')
    likes=train.loc[train.user_id.eq(user_id)&train.rating.ge(4),['user_id','recipe_id','rating']].merge(
        recipes[['recipe_id','name']],on='recipe_id')
    truth=test.loc[test.user_id.eq(user_id)&test.rating.ge(4)].merge(
        recipes[['recipe_id','name']],on='recipe_id')
    _,ids,scores=next(model.recommend([user_id],10,32))
    example=model.recipes.set_index('recipe_id').loc[ids,['name','minutes']].reset_index()
    example.insert(0,'user_id',user_id)
    example.insert(1,'rank',np.arange(1,len(example)+1))
    example['cosine_score']=scores
    example['test_positive_match']=example.recipe_id.isin(truth.recipe_id)
    if set(example.recipe_id)&set(train.loc[train.user_id.eq(user_id),'recipe_id']):
        raise AssertionError('Train에서 이미 평가한 레시피가 추천에 포함됐습니다.')
    if output_dir is not None:
        likes.to_csv(output_dir/'example_train_positives.csv',index=False,encoding='utf-8-sig')
        truth.to_csv(output_dir/'example_test_positives.csv',index=False,encoding='utf-8-sig')
        example.to_csv(output_dir/'example_top10.csv',index=False,encoding='utf-8-sig')
        save_json(output_dir/'example_user.json',dict(user_id=user_id,team_confirmed=user_id==187412,
            train_positive_count=len(likes),train_rated_count=int(train.user_id.eq(user_id).sum()),
            test_positive_count=len(truth),test_eligible_users=len(test_users),
            full_test_evaluation_performed=full_test_evaluation_performed))
    print('공통 예시 사용자:',user_id,'/ MF 담당자와 확정' if user_id==187412 else '/ 별도 사용자')
    print('Train 전체 평가 레시피 수:',int(train.user_id.eq(user_id).sum()))
    print('Train 4점 이상 레시피 수:',len(likes))
    print('Test 4점 이상 정답 레시피 수:',len(truth))
    print('Train 선호 레시피 예시:')
    print(likes[['recipe_id','rating','name']].head(5).to_string(index=False))
    print('Test 4점 이상 정답:')
    print(truth[['recipe_id','rating','name']].to_string(index=False))
    print('콘텐츠 모델 실제 Top-10:')
    print(example.to_string(index=False))
    return example

def evaluate_test_once(model, test, output_dir, selected_name, selected_weights):
    """이전 실행 경로 호환: 선택 완료 model의 test를 실제 평가하고 설정·행렬 해시를 기록."""
    print('Test 평가 시작:',selected_name,flush=True)
    started=time.perf_counter()
    metrics,details=model.evaluate(test,10,32)
    metrics.update(selected_name=selected_name,weights=selected_weights,
        evaluation_seconds=time.perf_counter()-started,
        users_with_hits=int(details.hits.gt(0).sum()))
    if output_dir is not None:
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        signature=dict(name=selected_name,weights=selected_weights,
            test_rows_sha256=hashlib.sha256(pd.util.hash_pandas_object(test,index=False).values.tobytes()).hexdigest(),
            matrix_sha256=hashlib.sha256(model.X.data.tobytes()+model.X.indices.tobytes()+model.X.indptr.tobytes()).hexdigest(),
            candidates_sha256=hashlib.sha256(model.ids.tobytes()).hexdigest(),
            preference_sha256=hashlib.sha256(model.W.data.tobytes()+model.W.indices.tobytes()+model.W.indptr.tobytes()).hexdigest())
        details.to_csv(output_dir/'test_per_user.csv',index=False)
        save_json(output_dir/'test_metrics.json',metrics)
        save_json(output_dir/'test_evaluation_manifest.json',signature)
    print('===== TEST 평가 결과 =====')
    print('선택 모델:',metrics['selected_name'])
    print('평가 사용자:',metrics['users'])
    print(f"Precision@10: {metrics['precision_at_10']:.12f}")
    print(f"Recall@10:    {metrics['recall_at_10']:.12f}")
    print('전체 적중수:',metrics['hits'])
    print('전체 정답 양성수:',metrics['ground_truth_positives'])
    return metrics


def plot_validation(comparison):
    """이전 2설정 재현용 그래프. 새 실험 그래프는 revision_plot이 F1까지 표시한다."""
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(comparison))
    ax.plot(x, comparison.precision_at_10, 'o-', color='royalblue', label='Precision@10 (Val)')
    ax.plot(x, comparison.recall_at_10, 's-', color='darkorange', label='Recall@10 (Val)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Ingredients 40% + Name 60%', 'Ingredients 50% + Name 30% + Tags 20%'])
    ax.set_xlabel('TF-IDF Configuration')
    ax.set_ylabel('Score')
    ax.set_title('Validation Performance by TF-IDF Configuration')
    ax.grid(True, linestyle='--', alpha=.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def validate_split(label, frame, train, recipes):
    """label 분할의 결측/평점/중복쌍/ID/정답 후보 포함 여부를 확인하고 오류 시 중단."""
    if frame[['user_id', 'recipe_id', 'rating']].isna().any().any() or frame.duplicated(['user_id', 'recipe_id']).any():
        raise ValueError(label + ': 결측 또는 사용자–레시피 중복')
    if not frame.rating.isin([1, 2, 3, 4, 5]).all() or not frame.recipe_id.isin(recipes.recipe_id).all():
        raise ValueError(label + ': 평점 범위 또는 레시피 ID 오류')
    if label == 'Train':
        return
    if len(pd.MultiIndex.from_frame(train[['user_id', 'recipe_id']]).intersection(pd.MultiIndex.from_frame(frame[['user_id', 'recipe_id']]))):
        raise ValueError('Train/' + label + ' 평가쌍 중복')
    if not frame.loc[frame.rating.ge(4), 'recipe_id'].isin(train.recipe_id).all():
        raise ValueError('Train 후보 밖의 ' + label + ' 양성 정답')


def run_pipeline(data_dir, output_dir=None, user_id=187412):
    """Recall 기준으로 두 설정을 비교하는 실험 재현 경로."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    required = ['recipe_features.csv', 'train_ratings.csv', 'validation_ratings.csv', 'test_ratings.csv']
    missing = [str(data_dir / name) for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    recipes = pd.read_csv(data_dir / 'recipe_features.csv')
    train = pd.read_csv(data_dir / 'train_ratings.csv')
    validation = pd.read_csv(data_dir / 'validation_ratings.csv')
    if recipes.recipe_id.duplicated().any() or recipes[['recipe_id', 'content_text']].isna().any().any():
        raise ValueError('레시피 ID 중복 또는 필수 특성 결측')
    validate_split('Train', train, train, recipes)
    validate_split('Validation', validation, train, recipes)
    base = ContentModel(recipes, train, max_features=50000, min_df=2, ngram_range=(1, 1), profile_weight='uniform')
    fields = build_fields(base.recipes, ['ingredients', 'name', 'tags'])
    print('Train 전체 레시피 후보 수:', len(base.ids), flush=True)
    print('Validation 전체 평가 사용자:', len(base.eligible(validation)), flush=True)
    records, models = [], {}
    for order, (name, weights) in enumerate(VALIDATION_CONFIGS):
        model = copy.copy(base)
        model.X = normalize(sparse.hstack([fields[field] * np.float32(np.sqrt(weight)) for field, weight in weights.items()], format='csr'))
        model.recommend = MethodType(fast_recommend, model)
        print('\nValidation 실제 평가 시작:', name, weights, flush=True)
        started = time.perf_counter()
        validation_metrics, _ = model.evaluate(validation, 10, 32)
        print(validation_metrics, flush=True)
        records.append(dict(name=name, order=order, weights=weights, **validation_metrics,
            evaluation_seconds=time.perf_counter()-started))
        models[name] = model
    comparison = pd.DataFrame(records)
    print('\n===== Validation 설정 비교 =====')
    print(comparison[['name', 'users', 'precision_at_10', 'recall_at_10', 'hits']].to_string(index=False))
    winner = comparison.sort_values(['recall_at_10', 'precision_at_10', 'order'], ascending=[False, False, True]).iloc[0]
    selected_model = models[winner['name']]
    print('Validation으로 선택한 설정:', winner['name'], winner['weights'], flush=True)
    plot_validation(comparison)
    if output_dir is not None:
        save_json(output_dir / 'selected_config.json', dict(name=winner['name'], weights=winner['weights'], selection_split='validation',
            validation_precision_at_10=float(winner['precision_at_10']), validation_recall_at_10=float(winner['recall_at_10'])))
    del fields, models, model, base
    test = pd.read_csv(data_dir / 'test_ratings.csv')
    validate_split('Test', test, train, recipes)
    metrics = evaluate_test_once(selected_model, test, output_dir, winner['name'], winner['weights'])
    export_team_example(selected_model, recipes, train, test, output_dir, user_id=user_id,
                        full_test_evaluation_performed=True)
    if output_dir is not None:
        print('결과 저장 폴더:', output_dir.resolve())
    return metrics


def run_example(data_dir, output_dir=None, user_id=187412):
    """과거 확정 모델로 한 사용자만 추천. 개선 모델 비교/전체 평가는 수행하지 않는다."""
    data_dir = Path(data_dir)
    recipes = pd.read_csv(data_dir / 'recipe_features.csv')
    train = pd.read_csv(data_dir / 'train_ratings.csv')
    if recipes.recipe_id.duplicated().any() or recipes[['recipe_id', 'content_text']].isna().any().any():
        raise ValueError('레시피 ID 중복 또는 필수 특성 결측')
    validate_split('Train', train, train, recipes)
    weights = next(weights for name, weights in VALIDATION_CONFIGS if name == FINAL_MODEL_NAME)
    model = ContentModel(recipes, train, max_features=50000, min_df=2, ngram_range=(1, 1), profile_weight='uniform')
    fields = build_fields(model.recipes, list(weights))
    model.X = normalize(sparse.hstack([fields[field] * np.float32(np.sqrt(weight)) for field, weight in weights.items()], format='csr'))
    model.recommend = MethodType(fast_recommend, model)
    print('예시 사용자만 추천합니다. 전체 Validation/Test 평가는 반복하지 않습니다.')
    print('기존 Validation에서 선택한 설정:', FINAL_MODEL_NAME, weights)
    print('Train 전체 레시피 후보 수:', len(model.ids))
    test = pd.read_csv(data_dir / 'test_ratings.csv')
    validate_split('Test', test, train, recipes)
    return export_team_example(model, recipes, train, test, output_dir, user_id=user_id)


# TF-IDF 설정 비교. Baseline의 후보/정답/제외 규칙은 그대로 사용한다.
# 이번 수정은 외부 블로그·GitHub 코드를 복사하지 않고 기존 프로젝트를 확장했다.
# 과거 코드의 외부 참고 이력은 확인된 기록 범위만 인정하며 출처를 추정하지 않는다.
# 라이브러리 API 참고(코드 복사 출처와 구분):
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
EXPERIMENTS = [
    dict(name='baseline', weights={'ingredients': .4, 'name': .6}, tf='original', rating=False, field_profile=False),
    dict(name='tags', weights={'ingredients': .5, 'name': .3, 'tags': .2}, tf='original', rating=False, field_profile=False),
    dict(name='ingredients70', weights={'ingredients': .7, 'name': .3}, tf='original', rating=False, field_profile=False),
    dict(name='rating', weights={'ingredients': .4, 'name': .6}, tf='original', rating=True, field_profile=False),
    dict(name='field_profiles', weights={'ingredients': .4, 'name': .6}, tf='original', rating=False, field_profile=True),
    dict(name='field_profiles_rating', weights={'ingredients': .4, 'name': .6}, tf='original', rating=True, field_profile=True),
    dict(name='binary_tf', weights={'ingredients': .4, 'name': .6}, tf='binary', rating=False, field_profile=False),
    dict(name='rare_unigrams', weights={'ingredients': .4, 'name': .6}, tf='rare', rating=False, field_profile=False),
]


def revision_fields(recipes, variant):
    """variant: 사전 고정한 TF 설정. Train 후보 텍스트만으로 어휘/IDF를 학습한다."""
    if variant == 'original':
        return build_fields(recipes, ['ingredients', 'name', 'tags'])
    matrices = {}
    for field, column, capacity in [('ingredients', 'ingredients_text', 20000), ('name', 'name', 60000)]:
        # binary: 출현 여부만 반영해 반복 단어 영향 제거.
        # rare: min_df=1로 희귀 레시피 어휘를 보존하고 이름은 unigram으로 비교.
        vectorizer = TfidfVectorizer(
            min_df=1 if variant == 'rare' else 2,
            max_df=1.0 if variant == 'rare' else (.95 if field == 'ingredients' else .8),
            max_features=capacity, ngram_range=(1, 1) if variant == 'rare' or field == 'ingredients' else (1, 2),
            sublinear_tf=False if variant == 'binary' else True,
            binary=variant == 'binary', token_pattern=r'(?u)\b\w+\b', dtype=np.float32)
        matrices[field] = vectorizer.fit_transform(recipes[column].fillna('')).tocsr()
    return matrices


def revision_recommend(self, users, k=10, batch_size=32):
    """특징별 프로필 L2 정규화: 재료/이름의 취향 집중도가 가중치를 왜곡하는 현상 완화."""
    for start in range(0, len(users), batch_size):
        block = users[start:start + batch_size]
        user_rows = self.users.get_indexer(block)
        scores = None
        for field, weight in self.config['weights'].items():
            matrix = self.fields[field]
            profile = normalize(self.W[user_rows] @ matrix)
            field_scores = np.ascontiguousarray((matrix @ profile.toarray().T).T)
            if scores is None:
                scores = field_scores * np.float32(weight)
            else:
                scores += field_scores * np.float32(weight)
        for offset, (user_id, row) in enumerate(zip(block, user_rows)):
            seen = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row + 1]]
            scores[offset, seen] = -np.inf
            selected = topk(scores[offset], self.ids, k)
            yield int(user_id), self.ids[selected], scores[offset, selected]


def configure_revision(base, fields, config):
    """base는 이력/후보를 공유하고 W/X만 설정별로 생성하여 중복 TF-IDF 학습을 막는다."""
    model = copy.copy(base)
    model.config = config
    model.fields = fields
    model.set_weight('rating' if config['rating'] else 'uniform')
    if config['field_profile']:
        model.recommend = MethodType(revision_recommend, model)
    else:
        # sqrt 가중치는 내적에서 weight가 된다. L2 정규화로 cosine을 계산한다.
        model.X = normalize(sparse.hstack([fields[field] * np.float32(np.sqrt(weight))
            for field, weight in config['weights'].items()], format='csr'))
        model.recommend = MethodType(fast_recommend, model)
    return model


def revision_plot(frame, path, title):
    """frame의 실측 P/R/F1을 path PNG에 저장하며 title은 분할/비교 조건을 표시한다.
    subplots(1,3)은 지표별 축 분리, figsize는 인치 크기, plot('o-')는 점+선이다.
    set_title/set_xlabel/set_ylabel은 제목/축 이름, set_xticks/set_xticklabels는 눈금/문자,
    tick_params(rotation=70)는 긴 설정명 회전, grid(alpha=.3)는 옅은 격자,
    ticklabel_format(scilimits=(0,0))는 과학 표기법, suptitle은 전체 제목이다.
    tight_layout은 겹침 완화, savefig(dpi=180,bbox_inches='tight')는 해상도/여백 지정,
    legend는 범례, show는 노트북 출력, close는 Figure 메모리 해제이다.
    """
    figure, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    for axis, metric in zip(axes, ['precision_at_10', 'recall_at_10', 'f1_at_10']):
        positions = np.arange(len(frame))
        axis.plot(positions, frame[metric], 'o-')
        axis.set_xticks(positions)
        axis.set_xticklabels(frame['name'], rotation=55, ha='right')
        axis.set_title(metric.replace('_at_', '@').upper())
        axis.grid(alpha=.3)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(figure)


def run_revision(data_dir, output_dir, user_id=187412):
    """전체 Validation으로 F1 우선 선택 후, 잠근 Baseline/선택 모델을 Test에서 각 1회 평가한다.

    data_dir: CSV 4개가 있는 폴더. output_dir: 지표/그래프/추천 예시 저장 폴더.
    user_id: 팀 공통 예시이며 모델 선정에는 사용하지 않는다.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes = pd.read_csv(data_dir / 'recipe_features.csv')
    train = pd.read_csv(data_dir / 'train_ratings.csv')
    validation = pd.read_csv(data_dir / 'validation_ratings.csv')
    validate_split('Train', train, train, recipes)
    validate_split('Validation', validation, train, recipes)
    base = ContentModel(recipes, train, 50000, 2, (1, 1), 'uniform')
    records = []
    field_cache = {}
    # TF 설정별 특징 행렬을 공유한다. Test에서도 재학습 없이 동일 행렬을 사용한다.
    for variant in ['original', 'binary', 'rare']:
        fields = revision_fields(base.recipes, variant)
        field_cache[variant] = fields
        for config in [entry for entry in EXPERIMENTS if entry['tf'] == variant]:
            print('\nVALIDATION', config, flush=True)
            model = configure_revision(base, fields, config)
            started = time.perf_counter()
            metrics, details = model.evaluate(validation)
            record = dict(name=config['name'], **metrics, seconds=time.perf_counter()-started)
            records.append(record)
            print(record, flush=True)
            details.to_csv(output_dir / (config['name'] + '_validation_users.csv'), index=False)
        del fields, model
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir / 'validation_comparison.csv', index=False)
    # F1 동점은 Recall→Precision→실험 순서. Test를 읽기 전에 선택을 저장한다.
    selected = comparison.sort_values(['f1_at_10', 'recall_at_10', 'precision_at_10'],
                                     ascending=False, kind='stable').iloc[0]['name']
    selected_config = next(config for config in EXPERIMENTS if config['name'] == selected)
    save_json(output_dir / 'selection_before_test.json', dict(selected=selected_config,
        criterion='macro F1@10, Recall@10, Precision@10', experiments=EXPERIMENTS))
    print('\n전체 Validation 실제 결과\n', comparison.to_string(index=False), flush=True)
    print('Validation F1 기준 선택:', selected_config, flush=True)
    revision_plot(comparison, output_dir / 'validation_comparison.png', 'Full Validation: fixed candidate universe')
    # Test는 모델 구성을 확정한 뒤 로드한다. Baseline 비교도 동일 평가 조건으로 수행한다.
    test = pd.read_csv(data_dir / 'test_ratings.csv')
    validate_split('Test', test, train, recipes)
    test_records = []
    for config in [EXPERIMENTS[0]] + ([] if selected == 'baseline' else [selected_config]):
        fields = field_cache[config['tf']]
        model = configure_revision(base, fields, config)
        metrics, details = model.evaluate(test)
        test_records.append(dict(name=config['name'], **metrics))
        details.to_csv(output_dir / (config['name'] + '_test_users.csv'), index=False)
        print('\nTEST', config['name'], metrics, flush=True)
        if config['name'] == selected:
            export_team_example(model, recipes, train, test, output_dir, user_id, True)
        del fields, model
    test_comparison = pd.DataFrame(test_records)
    test_comparison.to_csv(output_dir / 'test_comparison.csv', index=False)
    revision_plot(test_comparison, output_dir / 'test_comparison.png', 'Test: baseline vs validation-selected model')
    # 진단 통계는 원인 증명이 아닌 평가 난이도/희소성 설명을 위한 관측값이다.
    diagnostic = dict(candidate_count=len(base.ids), train_rows=len(train),
        train_users=len(base.users), density=len(train)/(len(base.users)*len(base.ids)),
        validation_users=len(base.eligible(validation)), test_users=len(base.eligible(test)),
        positive_history_median_validation=float(train[train.rating.ge(4)].groupby('user_id').size().reindex(base.eligible(validation)).median()),
        positive_history_median_test=float(train[train.rating.ge(4)].groupby('user_id').size().reindex(base.eligible(test)).median()),
        test_previously_observed=True)
    save_json(output_dir / 'diagnostics.json', diagnostic)
    return comparison, test_comparison


def latent_recommend(self, users, k=10, batch_size=32):
    """LSA 콘텐츠 벡터와 사용자 벡터의 cosine. 필요 시 최근 양성 3개의 최대 유사도를 혼합.
    latent는 레시피 메타데이터만 학습한다. Train 평점은 해당 사용자 프로필에만 사용한다.
    """
    for start in range(0, len(users), batch_size):
        block = users[start:start + batch_size]
        rows = self.users.get_indexer(block)
        profiles = normalize(self.W[rows] @ self.embedding)
        scores = np.ascontiguousarray(profiles @ self.embedding.T)
        for offset, (user_id, row) in enumerate(zip(block, rows)):
            if self.config['recent_mix']:
                # 최근 이력 선택은 사용자의 Train 날짜만 사용하며 추천 후보는 줄이지 않는다.
                recent = self.recent_items[int(user_id)]
                peak = (self.embedding[recent] @ self.embedding.T).max(axis=0)
                weight = self.config['recent_mix']
                scores[offset] = (1-weight)*scores[offset] + weight*peak
            seen = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row+1]]
            scores[offset, seen] = -np.inf
            selected = topk(scores[offset], self.ids, k)
            yield int(user_id), self.ids[selected], scores[offset, selected]


def residual_recommend(self, users, k=10, batch_size=32):
    """공통 텍스트 성분을 감쇠한 커널 cosine. K=XX^T-alpha*ZZ^T는 alpha<=1에서 양의 준정부호.
    item_norm으로 레시피를 먼저 정규화하고 Train 긍정 평균을 구성한다.
    거대한 잔차 특징/아이템 간 행렬을 생성하지 않고 원래 희소 내적과 저차원 내적의 차를 사용.
    """
    # projection은 X를 학습된 직교 텍스트 축으로 투영한 Z이다.
    # sum(axis=1)은 행별 제곱 길이, maximum(...,1e-8)은 반올림/영벡터로 인한 0 나눗셈 방지.
    projection = self.projection
    item_norm = np.sqrt(np.maximum(1-self.alpha*np.sum(projection*projection,axis=1),1e-8))
    # multiply는 후보별 열 스케일링이며 Train 양성 이력의 상대 가중치를 보정한다.
    preferences = self.W.multiply(1/item_norm).tocsr()
    for start in range(0,len(users),batch_size):
        block = users[start:start+batch_size]
        rows = self.users.get_indexer(block)
        raw_profile = preferences[rows] @ self.X
        small_profile = preferences[rows] @ projection
        norms = np.sqrt(np.maximum(np.asarray(raw_profile.multiply(raw_profile).sum(axis=1)).ravel()
                         -self.alpha*np.sum(small_profile*small_profile,axis=1),1e-8))
        scores = np.ascontiguousarray((self.X @ raw_profile.toarray().T).T)
        scores -= np.float32(self.alpha)*(small_profile @ projection.T)
        scores /= item_norm[None,:]
        scores /= norms[:,None]
        for offset,(uid,row) in enumerate(zip(block,rows)):
            seen = self.seen.indices[self.seen.indptr[row]:self.seen.indptr[row+1]]
            scores[offset,seen] = -np.inf
            selected = topk(scores[offset],self.ids,k)
            yield int(uid),self.ids[selected],scores[offset,selected]


def run_residual(data_dir,output_dir,user_id=187412):
    """data_dir 원본4CSV에서 공통 성분 차원/감쇠량을 비교해 output_dir에 실측 결과 저장.
    user_id는 출력용 공통 예시이며 모델 선택에 사용하지 않는다. 후보/평가 분할 변경 없음.
    차원8/32는 소수/다수의 공통 축 제거 비교, alpha .5/1은 부분/완전 감쇠 비교이다.
    """
    data_dir,output_dir=Path(data_dir),Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    recipes=pd.read_csv(data_dir/'recipe_features.csv')
    train=pd.read_csv(data_dir/'train_ratings.csv')
    validation=pd.read_csv(data_dir/'validation_ratings.csv')
    validate_split('Train',train,train,recipes)
    validate_split('Validation',validation,train,recipes)
    base=ContentModel(recipes,train,50000,2,(1,1),'uniform')
    fields=revision_fields(base.recipes,'rare')
    base=configure_revision(base,fields,EXPERIMENTS[-1])
    configurations=[dict(name='residual'+str(dim)+'_'+str(alpha),dimensions=dim,alpha=alpha)
                    for dim in [8,32] for alpha in [.5,1.0]]
    save_json(output_dir/'plan_before_validation.json',configurations)
    reducer=TruncatedSVD(n_components=32,n_iter=7,random_state=42)
    projected=reducer.fit_transform(base.X).astype(np.float32)
    records=[]
    for config in configurations:
        model=copy.copy(base)
        model.projection=np.ascontiguousarray(projected[:,:config['dimensions']])
        model.alpha=config['alpha']
        model.recommend=MethodType(residual_recommend,model)
        print('VALIDATION',config,flush=True)
        started=time.perf_counter()
        metrics,details=model.evaluate(validation)
        records.append(dict(name=config['name'],**metrics,seconds=time.perf_counter()-started))
        details.to_csv(output_dir/(config['name']+'_validation_users.csv'),index=False)
        print(records[-1],flush=True)
    comparison=pd.DataFrame(records)
    comparison.to_csv(output_dir/'validation_comparison.csv',index=False)
    winner=comparison.sort_values(['f1_at_10','recall_at_10','precision_at_10'],ascending=False,kind='stable').iloc[0]
    config=next(entry for entry in configurations if entry['name']==winner['name'])
    save_json(output_dir/'selection_before_test.json',dict(selected=config,validation=winner.to_dict()))
    revision_plot(comparison,output_dir/'validation_comparison.png','Common component correction: full Validation')
    print('Test 로드 전 선택:',config,flush=True)
    model=copy.copy(base)
    model.projection=np.ascontiguousarray(projected[:,:config['dimensions']])
    model.alpha=config['alpha']
    model.recommend=MethodType(residual_recommend,model)
    test=pd.read_csv(data_dir/'test_ratings.csv')
    validate_split('Test',test,train,recipes)
    metrics,details=model.evaluate(test)
    result=pd.DataFrame([dict(name=config['name'],**metrics)])
    result.to_csv(output_dir/'test_comparison.csv',index=False)
    details.to_csv(output_dir/(config['name']+'_test_users.csv'),index=False)
    print('TEST 평가 결과',metrics,flush=True)
    export_team_example(model,recipes,train,test,output_dir,user_id,True)
    return comparison,result


def run_advanced(data_dir, output_dir, user_id=187412):
    """추가 콘텐츠 모델: 사전 고정한 LSA 차원/프로필 비교 후 Validation F1으로만 선택.
    Test는 모든 Validation 계산과 선택 기록 저장이 끝난 뒤에만 읽는다.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipes = pd.read_csv(data_dir/'recipe_features.csv')
    train = pd.read_csv(data_dir/'train_ratings.csv')
    validation = pd.read_csv(data_dir/'validation_ratings.csv')
    validate_split('Train', train, train, recipes)
    validate_split('Validation', validation, train, recipes)
    base = ContentModel(recipes, train, 50000, 2, (1,1), 'uniform')
    fields = revision_fields(base.recipes, 'original')
    baseline = configure_revision(base, fields, EXPERIMENTS[0])
    # 한 번의 256차원 학습 결과를 잘라 낮은 차원 실험에도 재사용한다.
    reducer = TruncatedSVD(n_components=256, n_iter=7, random_state=42)
    print('LSA 텍스트 표현 학습 시작', flush=True)
    latent = reducer.fit_transform(baseline.X).astype(np.float32)
    print('LSA 설명 분산 합:', float(reducer.explained_variance_ratio_.sum()), flush=True)
    positive = train[train.rating.ge(4)].sort_values(['date','recipe_id'])
    recent = positive.groupby('user_id').tail(3).groupby('user_id').recipe_id.apply(list).to_dict()
    base.recent_items = {int(uid):base.item_index.get_indexer(ids) for uid,ids in recent.items()}
    configurations = [dict(name='lsa'+str(dim), dimensions=dim, rating=False, recent_mix=0)
                      for dim in [16,64,128,256]]
    configurations += [dict(name='lsa128_rating', dimensions=128, rating=True, recent_mix=0),
                       dict(name='lsa128_recent25', dimensions=128, rating=False, recent_mix=.25),
                       dict(name='lsa128_recent75', dimensions=128, rating=False, recent_mix=.75)]
    save_json(output_dir/'plan_before_validation.json', configurations)
    embeddings = {dim:normalize(np.ascontiguousarray(latent[:,:dim])) for dim in [16,64,128,256]}
    records = []
    for config in configurations:
        model = copy.copy(base)
        model.config = config
        model.embedding = embeddings[config['dimensions']]
        model.set_weight('rating' if config['rating'] else 'uniform')
        model.recommend = MethodType(latent_recommend, model)
        print('VALIDATION',config,flush=True)
        started = time.perf_counter()
        metrics, details = model.evaluate(validation)
        record = dict(name=config['name'], **metrics, seconds=time.perf_counter()-started)
        records.append(record)
        details.to_csv(output_dir/(config['name']+'_validation_users.csv'),index=False)
        print(record,flush=True)
    comparison = pd.DataFrame(records)
    comparison.to_csv(output_dir/'validation_comparison.csv',index=False)
    winner = comparison.sort_values(['f1_at_10','recall_at_10','precision_at_10'],ascending=False,kind='stable').iloc[0]
    selected_config = next(config for config in configurations if config['name']==winner['name'])
    save_json(output_dir/'selection_before_test.json',dict(selected=selected_config,validation=winner.to_dict(),
        note='선행 TF-IDF 실험과 별개의 콘텐츠 표현 실험군. Test로 구성 선택하지 않음.'))
    revision_plot(comparison,output_dir/'validation_comparison.png','LSA content models: full Validation')
    print('Test 로드 전 선택한 설정:',selected_config,flush=True)
    test = pd.read_csv(data_dir/'test_ratings.csv')
    validate_split('Test',test,train,recipes)
    model = copy.copy(base)
    model.config = selected_config
    model.embedding = embeddings[selected_config['dimensions']]
    model.set_weight('rating' if selected_config['rating'] else 'uniform')
    model.recommend = MethodType(latent_recommend,model)
    metrics,details = model.evaluate(test)
    test_result = pd.DataFrame([dict(name=selected_config['name'],**metrics)])
    test_result.to_csv(output_dir/'test_comparison.csv',index=False)
    details.to_csv(output_dir/(selected_config['name']+'_test_users.csv'),index=False)
    print('TEST 평가 결과', metrics,flush=True)
    export_team_example(model,recipes,train,test,output_dir,user_id,True)
    save_json(output_dir/'lsa_diagnostics.json',dict(explained_variance=float(reducer.explained_variance_ratio_.sum()),
        random_state=42,n_iter=7,training_matrix='Train recipe metadata TF-IDF, not user-item ratings'))
    return comparison,test_result


def run_submission_example(data_dir,output_dir=None,user_id=187412):
    """Validation에서 선택한 8성분 완전 감쇠 모델로 한 사용자만 추천. 전체 평가를 반복하지 않는다.
    차원32로 학습한 뒤 첫8축을 사용하는 실제 실험의 분해 조건도 동일하게 유지한다.
    """
    data_dir=Path(data_dir)
    recipes=pd.read_csv(data_dir/'recipe_features.csv')
    train=pd.read_csv(data_dir/'train_ratings.csv')
    validate_split('Train',train,train,recipes)
    model=ContentModel(recipes,train,50000,2,(1,1),'uniform')
    fields=revision_fields(model.recipes,'rare')
    model=configure_revision(model,fields,EXPERIMENTS[-1])
    reducer=TruncatedSVD(n_components=32,n_iter=7,random_state=42)
    model.projection=np.ascontiguousarray(reducer.fit_transform(model.X)[:,:8].astype(np.float32))
    model.alpha=1.0
    model.recommend=MethodType(residual_recommend,model)
    test=pd.read_csv(data_dir/'test_ratings.csv')
    validate_split('Test',test,train,recipes)
    return export_team_example(model,recipes,train,test,output_dir,user_id,False)


if __name__ == '__main__':
    # ArgumentParser는 CLI 규칙을 정의하고 add_argument는 옵션을 등록한다.
    # parse_args는 사용자 입력을 읽는다. default는 생략값, help는 도움말이다.
    parser = argparse.ArgumentParser(description='전체 후보 콘텐츠 추천: Validation Macro F1 선택 및 Test 비교')
    # Colab 업로드 위치 /content 또는 로컬의 전처리_완료를 기본 경로로 선택한다.
    local_data = Path(__file__).resolve().parent / '전처리_완료'
    parser.add_argument('--data-dir', type=Path, default=Path('/content') if Path('/content').is_dir() else local_data)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--user-id', type=int, default=187412)
    parser.add_argument('--example-only', action='store_true')
    # store_true는 인수 존재 시 True. Path는 경로, int는 사용자 ID 자료형을 검증한다.
    parser.add_argument('--revision', action='store_true', help='TF-IDF 8개 설정 비교')
    parser.add_argument('--legacy', action='store_true', help='과거 Recall 기반 2설정 재현')
    parser.add_argument('--advanced', action='store_true', help='LSA 콘텐츠 표현 추가 개선 실험')
    parser.add_argument('--residual', action='store_true', help='공통 콘텐츠 성분 감쇠 실험')
    args = parser.parse_args()
    if args.residual:
        run_residual(args.data_dir,args.output_dir or Path(__file__).parent/'outputs'/'residual_content',args.user_id)
    elif args.advanced:
        run_advanced(args.data_dir,args.output_dir or Path(__file__).parent/'outputs'/'advanced_content',args.user_id)
    elif args.example_only:
        run_submission_example(args.data_dir, args.output_dir, args.user_id)
    elif args.legacy:
        run_pipeline(args.data_dir, args.output_dir, args.user_id)
    elif args.revision:
        run_revision(args.data_dir, args.output_dir or Path(__file__).parent / 'outputs' / 'professor_revision', args.user_id)
    else:
        run_residual(args.data_dir,args.output_dir or Path(__file__).parent/'outputs'/'residual_content',args.user_id)
