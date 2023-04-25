import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 32,
    'n_estimators': 100000,
    'random_state': 123,
    'importance_type': 'gain',
}


def train_lgb(input_x, input_y, input_id, params, n_splits=5,):
    train_oof = np.zeros(len(input_x))
    metrics = []
    imp = pd.DataFrame()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    metrics = []
    for fold, (id_tr, id_va) in enumerate(skf.split(input_x, input_y)):
        print('-'*20 + f'{fold}' + '-'*20)
        tr_x, tr_y = input_x.iloc[id_tr], input_y.iloc[id_tr]
        va_x, va_y = input_x.iloc[id_va], input_y.iloc[id_va]

        model = lgb.LGBMClassifier(**params)
        model.fit(tr_x,
                  tr_y,
                  eval_set=(va_x, va_y),
                  early_stopping_rounds=100,
                  verbose=100,
                  )
        tr_y_proba = model.predict_proba(tr_x)[:, 1]
        va_y_proba = model.predict_proba(va_x)[:, 1]
        metric_tr = roc_auc_score(tr_y, tr_y_proba)
        metric_va = roc_auc_score(va_y, va_y_proba)
        metrics.append([fold, metric_tr, metric_va])
        print(f'auc {metric_tr} {metric_va}')

        train_oof[id_va] = va_y_proba

        _imp = pd.DataFrame(
            {'col': input_x.colums, 'imp': model.feature_importances_, 'nfold': fold})
        imp = pd.concat(imp, _imp)

    print('-'*30 + 'result' + '-'*30)

    metrics = np.array(metrics)
    print(metrics)
    print("[cv] tr:{:.4f}+-{:.4f}, va:{:.4f}+-{:.4f}".format(
        metrics[:, 1].mean(), metrics[:, 1].std(),
        metrics[:, 2].mean(), metrics[:, 2].std(),
    ))
    print("[oof] {:.4f}".format(
        roc_auc_score(input_y, train_oof)
    ))

    train_oof = pd.concat([
        input_id,
        pd.DataFrame({"pred": train_oof})
    ], axis=1)

    imp = imp.groupby("col")["imp"].agg(
        ["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    return train_oof, imp, metrics
