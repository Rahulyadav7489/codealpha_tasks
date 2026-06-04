from xgboost import XGBClassifier
from split import X_train, y_train

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42
)

model.fit(X_train, y_train)