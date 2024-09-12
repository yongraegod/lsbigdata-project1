from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimatiors=50,
                                  max_samples=100,
                                   n_jobs=-1, random_state=42)

# n_estimatiors: Bagging에 사용될 모델의 갯수
# max_samples: 데이터셋 만들때 뽑은 표본 크기
# n_jobs: 시스템의 모든 가용 CPU 코어를 사용하여 병렬 처리를 수행

# bagging_model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(n_estimators=50,
                                max_leaf_nodes=16,
                                n_jobs=-1, random_state=42)
# rf_model.fit(X_train, y_train)