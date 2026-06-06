from sklearn.ensemble import RandomForestClassifier
from split import X_train, y_train


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)