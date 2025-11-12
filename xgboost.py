# ===============================================================
# CREDIT SCORING - DOMAIN-INFORMED FEATURE ENGINEERING
# TARGET: 0.68-0.70 ROC-AUC
# ===============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ---------- Load Data ----------
train = pd.read_csv("/kaggle/input/aiful-dataset/train.csv")
test = pd.read_csv("/kaggle/input/aiful-dataset/test.csv")
sample_submission = pd.read_csv("/kaggle/input/aiful-dataset/sample_submission.csv")
TARGET = "Default 12 Flag"

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target distribution: {train[TARGET].value_counts(normalize=True)}")

# ---------- DOMAIN-INFORMED CATEGORICAL ENCODINGS ----------
def create_domain_features(df):
    """Create features based on domain knowledge from metadata"""
    df = df.copy()
    
    # === MARKETING CHANNEL RISK ===
    # Digital channels (11=Internet) are typically different risk profile
    df['IS_INTERNET'] = (df['Major Media Code'] == 11).astype(int)
    df['IS_TRADITIONAL_MEDIA'] = df['Major Media Code'].isin([6, 7, 8]).astype(int)  # TV, Newspaper, Magazine
    df['IS_PHYSICAL_CHANNEL'] = df['Major Media Code'].isin([1, 2, 3]).astype(int)  # Introduction, Phone, Store
    
    # Internet channel sophistication
    df['IS_ORGANIC_SEARCH'] = ((df['Major Media Code'] == 11) & (df['Internet Details'] == 1)).astype(int)
    df['IS_PAID_ADVERTISING'] = ((df['Major Media Code'] == 11) & (df['Internet Details'].isin([2, 4]))).astype(int)
    
    # === APPLICATION CHANNEL RISK ===
    df['IS_MOBILE_APP'] = df['Reception Type Category'].isin([1701, 1801]).astype(int)  # Mobile applications
    df['IS_DIGITAL_CHANNEL'] = df['Reception Type Category'].isin([502, 1701, 1801]).astype(int)  # PC + Mobile
    df['IS_IN_PERSON'] = df['Reception Type Category'].isin([0, 1]).astype(int)  # In-store, Contract Room
    df['IS_CALL_CENTER'] = (df['Reception Type Category'] == 101).astype(int)
    
    # === EMPLOYMENT STABILITY ===
    # President/Employee (1,2) are more stable than contract/part-time (3,4,5)
    df['IS_PERMANENT_EMPLOYEE'] = df['Employment Type'].isin([1, 2]).astype(int)
    df['IS_CONTRACT_WORKER'] = df['Employment Type'].isin([3, 4, 5]).astype(int)
    
    # Employment status (1=Regular is most stable)
    df['IS_REGULAR_EMPLOYMENT'] = (df['Employment Status Type'] == 1).astype(int)
    
    # === INDUSTRY RISK PROFILES ===
    # Stable industries: Financial, Government, Public
    df['IS_STABLE_INDUSTRY'] = df['Industry Type'].isin([2, 3, 4, 15, 16, 17]).astype(int)  
    # Financial, Securities, Insurance, Hospital, School, Government
    
    # High-risk industries: Student, Agriculture, Others
    df['IS_HIGH_RISK_INDUSTRY'] = df['Industry Type'].isin([19, 18, 99]).astype(int)
    
    # === COMPANY SIZE STABILITY ===
    # Public employee and listed companies are most stable
    df['IS_PUBLIC_LISTED'] = df['Company Size Category'].isin([1, 2]).astype(int)
    df['IS_LARGE_COMPANY'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)  # 100+ employees
    df['IS_SMALL_COMPANY'] = df['Company Size Category'].isin([7, 8, 9]).astype(int)  # <20 employees
    
    # === RESIDENCE STABILITY ===
    # Own home (with/without loan) is more stable than rental
    df['IS_HOME_OWNER'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['IS_RENTER'] = df['Residence Type'].isin([4, 5]).astype(int)
    df['IS_DORMITORY'] = df['Residence Type'].isin([6, 7]).astype(int)
    df['HAS_HOME_LOAN'] = df['Residence Type'].isin([2, 9]).astype(int)
    
    # === FAMILY STRUCTURE ===
    df['IS_MARRIED'] = (df['Single/Married Status'] == 2).astype(int)
    df['HAS_DEPENDENTS'] = (df['Number of Dependents'] > 0).astype(int)
    df['IS_SINGLE_LIVING_ALONE'] = (df['Family Composition Type'] == 5).astype(int)
    df['HAS_SPOUSE_AND_CHILDREN'] = df['Family Composition Type'].isin([2, 3]).astype(int)
    
    # === INSURANCE TYPE (Social Insurance is better than National) ===
    df['HAS_COMPANY_INSURANCE'] = df['Insurance Job Type'].isin([1, 3]).astype(int)
    df['HAS_SELF_EMPLOYED_INSURANCE'] = df['Insurance Job Type'].isin([2, 4]).astype(int)
    
    return df

train = create_domain_features(train)
test = create_domain_features(test)

# ---------- TEMPORAL FEATURES ----------
def create_temporal_features(df):
    df = df.copy()
    
    # Missing indicators
    df['JIS_Address_Missing'] = (df['JIS Address Code'].isna()).astype(int)
    df['JIS Address Code'] = df['JIS Address Code'].fillna(-999).astype(int)
    
    # Parse dates
    df['Application Date'] = pd.to_datetime(df['Application Date'])
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'])
    
    # Time of application
    df['Application_Hour'] = df['Application Time'] // 10000
    df['Application_Minute'] = (df['Application Time'] % 10000) // 100
    
    # Business hours vs off-hours (risk indicator)
    df['IS_BUSINESS_HOURS'] = ((df['Application_Hour'] >= 9) & (df['Application_Hour'] <= 17)).astype(int)
    df['IS_LATE_NIGHT'] = ((df['Application_Hour'] >= 22) | (df['Application_Hour'] <= 5)).astype(int)
    df['IS_EARLY_MORNING'] = ((df['Application_Hour'] >= 6) & (df['Application_Hour'] <= 8)).astype(int)
    
    # Date features
    df['Application_Month'] = df['Application Date'].dt.month
    df['Application_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['Application_Day'] = df['Application Date'].dt.day
    df['Application_Quarter'] = df['Application Date'].dt.quarter
    df['Application_IsWeekend'] = (df['Application_DayOfWeek'] >= 5).astype(int)
    df['Application_IsMonthEnd'] = (df['Application_Day'] > 25).astype(int)
    df['Application_IsMonthStart'] = (df['Application_Day'] <= 5).astype(int)
    
    # Age calculation
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['Birth_Year'] = df['Date of Birth'].dt.year
    df['Birth_Month'] = df['Date of Birth'].dt.month
    
    return df

train = create_temporal_features(train)
test = create_temporal_features(test)

# ---------- FINANCIAL FEATURES ----------
epsilon = 1e-6

def winsorize(series, lower=0.01, upper=0.99):
    return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))

def create_financial_features(df):
    df = df.copy()
    
    # === INCOME FEATURES ===
    df['MONTHLY_INCOME'] = df['Total Annual Income'] / 12
    df['LOG_INCOME'] = np.log1p(df['Total Annual Income'])
    
    # === LOAN RATIOS ===
    df['LOAN_TO_INCOME'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + epsilon)
    df['LOAN_TO_MONTHLY'] = df['Application Limit Amount(Desired)'] / (df['MONTHLY_INCOME'] + epsilon)
    
    # === DEBT RATIOS ===
    df['DECLARED_DEBT_TO_INCOME'] = df['Declared Amount of Unsecured Loans'] / (df['Total Annual Income'] + epsilon)
    df['ACTUAL_DEBT_TO_INCOME'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + epsilon)
    df['RENT_TO_INCOME'] = df['Rent Burden Amount'] / (df['MONTHLY_INCOME'] + epsilon)
    
    # Winsorize ratios
    for col in ['LOAN_TO_INCOME', 'DECLARED_DEBT_TO_INCOME', 'ACTUAL_DEBT_TO_INCOME', 'RENT_TO_INCOME']:
        df[col] = winsorize(df[col])
    
    # === DEBT HONESTY INDICATOR ===
    df['DEBT_DECLARED_VS_ACTUAL'] = df['Declared Amount of Unsecured Loans'] / (df['Amount of Unsecured Loans'] + epsilon)
    df['DEBT_UNDERREPORTED'] = (df['Amount of Unsecured Loans'] > df['Declared Amount of Unsecured Loans'] + 10000).astype(int)
    df['DEBT_OVERREPORTED'] = (df['Declared Amount of Unsecured Loans'] > df['Amount of Unsecured Loans'] + 10000).astype(int)
    df['DEBT_DIFF_ABS'] = np.abs(df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans'])
    df['DEBT_DIFF_RATIO'] = df['DEBT_DIFF_ABS'] / (df['Amount of Unsecured Loans'] + 1000)
    
    # === LOAN COUNT FEATURES ===
    df['HAS_NO_LOANS'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    df['HAS_MANY_LOANS'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['AVG_LOAN_SIZE'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['LOAN_COUNT_BUCKET'] = pd.cut(df['Number of Unsecured Loans'], bins=[-1, 0, 1, 2, 100], labels=[0,1,2,3]).astype(int)
    
    # === EMPLOYMENT & STABILITY ===
    df['EMPLOYMENT_YEARS'] = df['Duration of Employment at Company (Months)'] / 12
    df['LOG_EMPLOYMENT_YEARS'] = np.log1p(df['EMPLOYMENT_YEARS'])
    df['STABILITY_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['Age'] + epsilon)
    df['IS_NEW_EMPLOYEE'] = (df['EMPLOYMENT_YEARS'] < 1).astype(int)
    df['IS_LONG_TERM_EMPLOYEE'] = (df['EMPLOYMENT_YEARS'] >= 5).astype(int)
    
    # === DEPENDENT ECONOMICS ===
    df['INCOME_PER_DEPENDENT'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    df['INCOME_PER_FAMILY_MEMBER'] = df['Total Annual Income'] / (df['Number of Dependents'] + 2)  # +2 for applicant + spouse
    df['HAS_HIGH_DEPENDENT_BURDEN'] = (df['Number of Dependents'] >= 3).astype(int)
    
    # === LIQUIDITY & BURDEN ===
    df['TOTAL_FIXED_OBLIGATIONS'] = df['Rent Burden Amount'] + df['Amount of Unsecured Loans']
    df['MONTHLY_OBLIGATIONS'] = df['TOTAL_FIXED_OBLIGATIONS'] / 12
    df['OBLIGATIONS_TO_INCOME'] = df['TOTAL_FIXED_OBLIGATIONS'] / (df['Total Annual Income'] + epsilon)
    df['FREE_INCOME_MONTHLY'] = df['MONTHLY_INCOME'] - (df['MONTHLY_OBLIGATIONS'])
    df['FREE_INCOME_RATIO'] = df['FREE_INCOME_MONTHLY'] / (df['MONTHLY_INCOME'] + epsilon)
    
    # === COMPOSITE RISK SCORES ===
    df['FINANCIAL_STRESS_SCORE'] = (
        df['ACTUAL_DEBT_TO_INCOME'] * 0.35 + 
        df['LOAN_TO_INCOME'] * 0.25 + 
        df['RENT_TO_INCOME'] * 0.20 +
        df['OBLIGATIONS_TO_INCOME'] * 0.20
    )
    
    df['STABILITY_SCORE'] = (
        df['IS_PERMANENT_EMPLOYEE'] * 0.3 +
        df['IS_HOME_OWNER'] * 0.3 +
        df['STABILITY_RATIO'] * 0.2 +
        df['IS_LARGE_COMPANY'] * 0.2
    )
    
    # === AGE BUCKETS ===
    df['AGE_BUCKET'] = pd.cut(df['Age'], bins=[0,25,30,35,40,45,50,55,100], 
                               labels=[0,1,2,3,4,5,6,7]).astype(int)
    df['IS_YOUNG_APPLICANT'] = (df['Age'] < 25).astype(int)
    df['IS_SENIOR_APPLICANT'] = (df['Age'] > 55).astype(int)
    
    # === INCOME BUCKETS ===
    df['INCOME_BUCKET'] = pd.qcut(df['Total Annual Income'], 10, labels=False, duplicates='drop')
    df['DEBT_BUCKET'] = pd.qcut(df['ACTUAL_DEBT_TO_INCOME'], 10, labels=False, duplicates='drop')
    df['EMPLOYMENT_BUCKET'] = pd.cut(df['EMPLOYMENT_YEARS'], bins=[-1,1,2,3,5,10,100], 
                                      labels=[0,1,2,3,4,5]).astype(int)
    
    # === POLYNOMIAL FEATURES ===
    df['DEBT_INCOME_SQ'] = df['ACTUAL_DEBT_TO_INCOME'] ** 2
    df['LOAN_INCOME_SQ'] = df['LOAN_TO_INCOME'] ** 2
    df['AGE_SQ'] = df['Age'] ** 2
    df['EMPLOYMENT_SQ'] = df['EMPLOYMENT_YEARS'] ** 2
    
    # === INTERACTION FEATURES ===
    df['DEBT_X_AGE'] = df['ACTUAL_DEBT_TO_INCOME'] * df['Age']
    df['DEBT_X_EMPLOYMENT'] = df['ACTUAL_DEBT_TO_INCOME'] * df['EMPLOYMENT_YEARS']
    df['INCOME_X_AGE'] = df['Total Annual Income'] * df['Age'] / 1e6
    df['INCOME_X_EMPLOYMENT'] = df['Total Annual Income'] * df['EMPLOYMENT_YEARS'] / 1e6
    df['AGE_X_EMPLOYMENT'] = df['Age'] * df['EMPLOYMENT_YEARS']
    df['LOAN_X_DEBT'] = df['LOAN_TO_INCOME'] * df['ACTUAL_DEBT_TO_INCOME']
    
    # Risk combinations
    df['HIGH_DEBT_LOW_INCOME'] = ((df['ACTUAL_DEBT_TO_INCOME'] > 0.5) & (df['INCOME_BUCKET'] < 3)).astype(int)
    df['LOW_STABILITY_HIGH_DEBT'] = ((df['STABILITY_RATIO'] < 0.1) & (df['ACTUAL_DEBT_TO_INCOME'] > 0.3)).astype(int)
    df['CONTRACT_WORKER_HIGH_DEBT'] = (df['IS_CONTRACT_WORKER'] & (df['ACTUAL_DEBT_TO_INCOME'] > 0.3)).astype(int)
    
    return df

train = create_financial_features(train)
test = create_financial_features(test)

# ---------- LABEL ENCODING ----------
cat_cols_basic = [
    'Major Media Code', 'Internet Details', 'Reception Type Category', 'Gender',
    'Single/Married Status', 'Residence Type', 'Employment Type', 'Employment Status Type',
    'Industry Type', 'Company Size Category', 'Name Type', 'Family Composition Type',
    'Living Arrangement Type', 'Insurance Job Type', 'AGE_BUCKET', 'EMPLOYMENT_BUCKET',
    'INCOME_BUCKET', 'DEBT_BUCKET', 'LOAN_COUNT_BUCKET'
]

for col in cat_cols_basic:
    if col in train.columns:
        le = LabelEncoder()
        full = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(full)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# ---------- TARGET ENCODING (CV-based) ----------
def target_encode_cv(train_df, test_df, cat_cols, target_col, n_splits=5):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for col in cat_cols:
        global_mean = train_df[target_col].mean()
        train_encoded[f'{col}_TE'] = global_mean
        
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr_idx, val_idx in kf.split(train_df, train_df[target_col]):
            means = train_df.iloc[tr_idx].groupby(col)[target_col].mean()
            train_encoded.loc[val_idx, f'{col}_TE'] = train_df.iloc[val_idx][col].map(means).fillna(global_mean)
        
        means = train_df.groupby(col)[target_col].mean()
        test_encoded[f'{col}_TE'] = test_df[col].map(means).fillna(global_mean)
    
    return train_encoded, test_encoded

te_cols = ['Industry Type', 'Employment Type', 'JIS Address Code', 'Company Size Category',
           'Major Media Code', 'Reception Type Category', 'Residence Type']
train, test = target_encode_cv(train, test, te_cols, TARGET)

# ---------- GROUP AGGREGATION FEATURES (CV-based) ----------
def create_group_features_cv(train_df, test_df, group_cols, agg_cols, target_col, n_splits=5):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    for gcol in group_cols:
        for acol in agg_cols:
            # Initialize
            train_encoded[f'{acol}_{gcol}_mean'] = 0
            train_encoded[f'{acol}_{gcol}_std'] = 0
            
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for tr_idx, val_idx in kf.split(train_df, train_df[target_col]):
                stats_mean = train_df.iloc[tr_idx].groupby(gcol)[acol].mean()
                stats_std = train_df.iloc[tr_idx].groupby(gcol)[acol].std().fillna(0)
                
                train_encoded.loc[val_idx, f'{acol}_{gcol}_mean'] = train_df.iloc[val_idx][gcol].map(stats_mean)
                train_encoded.loc[val_idx, f'{acol}_{gcol}_std'] = train_df.iloc[val_idx][gcol].map(stats_std)
            
            train_encoded[f'{acol}_{gcol}_mean'].fillna(train_df[acol].mean(), inplace=True)
            train_encoded[f'{acol}_{gcol}_std'].fillna(0, inplace=True)
            
            # For test
            stats_mean = train_df.groupby(gcol)[acol].mean()
            stats_std = train_df.groupby(gcol)[acol].std().fillna(0)
            test_encoded[f'{acol}_{gcol}_mean'] = test_df[gcol].map(stats_mean).fillna(train_df[acol].mean())
            test_encoded[f'{acol}_{gcol}_std'] = test_df[gcol].map(stats_std).fillna(0)
            
            # Relative features
            train_encoded[f'{acol}_vs_{gcol}_mean'] = train_encoded[acol] / (train_encoded[f'{acol}_{gcol}_mean'] + epsilon)
            test_encoded[f'{acol}_vs_{gcol}_mean'] = test_encoded[acol] / (test_encoded[f'{acol}_{gcol}_mean'] + epsilon)
    
    return train_encoded, test_encoded

group_cols = ['Industry Type', 'Employment Type', 'Company Size Category', 'Residence Type', 
              'AGE_BUCKET', 'INCOME_BUCKET']
agg_cols = ['Total Annual Income', 'ACTUAL_DEBT_TO_INCOME', 'LOAN_TO_INCOME', 
            'Application Limit Amount(Desired)', 'FINANCIAL_STRESS_SCORE']

train, test = create_group_features_cv(train, test, group_cols, agg_cols, TARGET)

# ---------- FEATURE SELECTION ----------
drop_cols = [TARGET, 'Application Date', 'Date of Birth', 'Application Time', 'Birth_Year']
features = [c for c in train.columns if c not in drop_cols]

X, y = train[features], train[TARGET]
X_test = test[features]

print(f"\n✅ Total features: {len(features)}")
print(f"✅ Train shape: {X.shape}, Test shape: {X_test.shape}")

# ---------- MODEL TRAINING ----------
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
sub_xgb = np.zeros(len(X_test))
sub_lgb = np.zeros(len(X_test))

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 20,
    'subsample': 0.74,
    'colsample_bytree': 0.79,
    'reg_lambda': 1.14,
    'reg_alpha': 0.35,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'random_state': 42,
}

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 20,
    'subsample': 0.75,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.3,
    'random_state': 42,
    'verbosity': -1,
    'device': 'gpu'
}

for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/5")
    print(f"{'='*60}")
    
    X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    # XGBoost
    print("\n🔷 Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=500
    )
    
    oof_xgb[val_idx] = xgb_model.predict(dval)
    sub_xgb += xgb_model.predict(dtest) / folds.n_splits
    
    # LightGBM
    print("\n🔶 Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=500)
        ]
    )
    
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    sub_lgb += lgb_model.predict(X_test) / folds.n_splits
    
    fold_auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
    fold_auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_auc_blend = roc_auc_score(y_val, 0.5 * oof_xgb[val_idx] + 0.5 * oof_lgb[val_idx])
    
    print(f"\n📊 Fold {fold+1} Results:")
    print(f"   XGBoost AUC: {fold_auc_xgb:.4f}")
    print(f"   LightGBM AUC: {fold_auc_lgb:.4f}")
    print(f"   Blend AUC: {fold_auc_blend:.4f}")

# Final ensemble
oof_ensemble = 0.5 * oof_xgb + 0.5 * oof_lgb
sub_ensemble = 0.5 * sub_xgb + 0.5 * sub_lgb

auc_xgb = roc_auc_score(y, oof_xgb)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_ensemble = roc_auc_score(y, oof_ensemble)

print(f"\n{'='*60}")
print(f"🎯 FINAL RESULTS")
print(f"{'='*60}")
print(f"XGBoost OOF AUC:   {auc_xgb:.6f}")
print(f"LightGBM OOF AUC:  {auc_lgb:.6f}")
print(f"Ensemble OOF AUC:  {auc_ensemble:.6f}")
print(f"{'='*60}")

# Save submissions
sample_submission[TARGET] = sub_ensemble
sample_submission.to_csv("submission_domain_informed.csv", index=False)
print("\n✅ Submission saved: submission_domain_informed.csv")

# Also save individual model predictions for further ensembling
sample_submission[TARGET] = sub_xgb
sample_submission.to_csv("submission_xgb_only.csv", index=False)
sample_submission[TARGET] = sub_lgb
sample_submission.to_csv("submission_lgb_only.csv", index=False)
print("✅ Individual model submissions also saved")
