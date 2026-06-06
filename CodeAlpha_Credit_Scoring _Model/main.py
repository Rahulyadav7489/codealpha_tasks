from split import X_test, y_test, X_train, y_train, scaler

from xg_boost import model as xg
from random_forest import model as rf
from logistic_regression import model as lr

import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

models= [xg, rf, lr]

for model in models:
    from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
    )

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:,1]
    print(f"for model {model}")
    print("Accuracy:", accuracy_score(y_test,pred))
    print("AUC:", roc_auc_score(y_test,prob))
    # print(classification_report(y_test,pred))

    train_prob = model.predict_proba(X_train)[:,1]
    test_prob = model.predict_proba(X_test)[:,1]

    print("Train AUC:", roc_auc_score(y_train, train_prob))
    print("Test AUC:", roc_auc_score(y_test, test_prob))

def find_credit_score(new_customer, model):
    new_df = pd.DataFrame([new_customer])

    new_df_scaled = scaler.transform(new_df)

    pred = model.predict(new_df_scaled)[0]

    predict = "good" if pred == 1 else "bad"

    prob = model.predict_proba(new_df_scaled)
    good_prob = prob[:, 1]

    credit_score = int(300 + good_prob[0] * 600)

    return {
        "prediction": predict,
        "credit_score": credit_score
    }


joblib.dump(rf, "credit_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")