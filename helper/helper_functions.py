import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import optuna.visualization as vis
import re
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool, cpu_count
from imblearn.over_sampling import SMOTE


random_state = 42

state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'Puerto Rico': 'PR'
}

binning_config = {
    'credit_score': (
        [299, 550, 660, 700, 725, 740, 750, 790, 850],
        ['(299, 550]', '(550, 660]', '(660, 700]', '(700, 725]', '(725, 740]', '(740, 750]', '(750, 790]', '(790, 850]']
    ),
    'original_debt_to_income_ratio': (
        [0, 23.5, 27.5, 29.5, 35.5, 42.5, 45.5, 50, 999],
        ['(0, 23.5]', '(23.5, 27.5]', '(27.5, 29.5]', '(29.5, 35.5]', '(35.5, 42.5]', '(42.5, 45.5]', '(45.5, 50]', '(998, 999]']
    ),
    'original_loan_to_value': (
        [0, 44.5, 50.5, 70.5, 89.5, 90.5, 109.5, 1000],
        ['(0, 44.5]', '(44.5, 50.5]', '(50.5, 70.5]', '(70.5, 89.5]', '(89.5, 90.5]', '(90.5, 109.5]', '(109.5, 1000]']
    ),
    'original_loan_term': (
        [0, 180, 500],
        ['(0, 180]', '(180, 500]']
    ),
    'original_upb': (
        [0, 98000, 1e10],
        ['(0, 98000]', '(98000, 1e10]']
    )
}

categorical_features_ml = [
    'first_time_homebuyer_flag', 
    'occupancy_status', 
    # 'property_type', 
    'loan_purpose', 
    # 'servicer_name_group', 
    'property_state',
    # 'credit_score_bins',
    # 'original_loan_to_value_bins',
    # 'original_debt_to_income_ratio_bins',
    # 'original_upb_bins',
    # 'original_loan_term_bins'
    ]
numerical_features_ml = [
        'borrowers_times_credit_score',
        'sato_f30',
        'zato',
        'credit_score',
        'original_debt_to_income_ratio',
        'original_upb',
        'original_loan_term',
        'original_loan_to_value', 
        'interest_diff_percentage',
        # 'number_of_units',
        'number_of_borrowers',
        'index_sa_state_mom12', 
        'State Unemployment Rate', 
        # 'National Unemployment Rate', 
        # 'CPIAUCSL'
        'credit_score_times_debt_to_income_ratio',
        'credit_score_times_loan_to_value'
       ]


def process_yearly_data(df_orig, monthly_avg_rate):
    df_orig = df_orig.merge(monthly_avg_rate, left_on='first_payment_date', right_on='Date', how='left')

    statistics = df_orig.groupby('first_payment_date').agg(
    original_interest_rate_mean=('original_interest_rate', 'mean'),
    original_interest_rate_std=('original_interest_rate', 'std'),
    count=('original_interest_rate', 'count')
    ).ffill()

    df_orig = df_orig.merge(statistics, left_on='first_payment_date', right_index=True, how='left')


    df_orig['sato_f30'] = df_orig['original_interest_rate'] - df_orig['U.S. 30 yr FRM']
    df_orig['zato'] = (df_orig['original_interest_rate'] - df_orig['original_interest_rate_mean']) / df_orig['original_interest_rate_std']
    
    df_orig['average_interest_rate'] = df_orig.apply(
        lambda row: row['U.S. 30 yr FRM'] if row['30yrFRM'] else row['U.S. 15 yr FRM'],
        axis=1
    )
    
    df_orig.drop(columns=['Date', 'original_interest_rate_mean', 'original_interest_rate_std', 'count'], inplace=True)
    
    df_orig['interest_diff_percentage'] = (
        (df_orig['original_interest_rate'] - df_orig['average_interest_rate'])
        / df_orig['average_interest_rate']
    )


    return df_orig


def process_loan_data(df_orig, df_svcg):
    df_orig = df_orig[~df_orig['property_state'].isin(['GU', 'PR', 'VI'])]

    loan_history = df_svcg.groupby(by='loan_sequence_number').agg(history=('loan_age', 'count')).reset_index().sort_values(by='history')
    exclude_list = loan_history.loc[loan_history['history'] <= 6, 'loan_sequence_number']
    df_orig = df_orig[~df_orig['loan_sequence_number'].isin(exclude_list)]

    df_merged = pd.merge(df_svcg, df_orig, how='left', on='loan_sequence_number')

    df_merged['real_loan_age'] = (
        (df_merged['monthly_reporting_period'].dt.year - df_merged['first_payment_date'].dt.year) * 12 +
        (df_merged['monthly_reporting_period'].dt.month - df_merged['first_payment_date'].dt.month) + 1
    )

    conditions_60 = (df_merged['modification_flag'] == 'Y') | (~df_merged['current_loan_delinquency_status'].isin(['0', '1']))
    conditions_90 = (df_merged['modification_flag'] == 'Y') | (~df_merged['current_loan_delinquency_status'].isin(['0', '1', '2']))

    loan_sequence_everD60 = df_merged.loc[conditions_60].groupby('loan_sequence_number', as_index=False).agg(
        everD60_date=('monthly_reporting_period', 'min'),
        everD60_age=('real_loan_age', 'min')
    )
    loan_sequence_everD90 = df_merged.loc[conditions_90].groupby('loan_sequence_number', as_index=False).agg(
        everD90_date=('monthly_reporting_period', 'min'),
        everD90_age=('real_loan_age', 'min')
    )

    loan_sequence_everDX = loan_sequence_everD60.merge(loan_sequence_everD90, how='left', on='loan_sequence_number')

    loan_sequence_everDX['ever_D60_3years_flag'] = loan_sequence_everDX['everD60_age'] <= (3 * 12)
    loan_sequence_everDX['ever_D90_3years_flag'] = loan_sequence_everDX['everD90_age'] <= (3 * 12)

    df_orig = df_orig.merge(
        loan_sequence_everDX[['loan_sequence_number', 'ever_D60_3years_flag', 'ever_D90_3years_flag']],
        how='left',
        on='loan_sequence_number'
    )

    df_orig['ever_D60_3years_flag'] = np.where(df_orig['ever_D60_3years_flag'] == True, 1, 0)
    df_orig['ever_D90_3years_flag'] = np.where(df_orig['ever_D90_3years_flag'] == True, 1, 0)


    df_orig = df_orig[df_orig['credit_score'] != 9999]

    df_orig['servicer_name_group'] = np.where(df_orig['servicer_name'] == 'Other servicers', 0, 1)

    df_orig['credit_score_times_debt_to_income_ratio'] = df_orig['credit_score'] * df_orig['original_debt_to_income_ratio']
    df_orig['credit_score_times_loan_to_value'] = df_orig['credit_score'] * df_orig['original_loan_to_value']
    df_orig['borrowers_times_credit_score'] = df_orig['number_of_borrowers'] * df_orig['credit_score']

    return df_orig

def merge_data(df_orig, df_hpi, df_unemp_state, df_unemp_national, df_cpi):
    df_hpi = df_hpi.sort_values(by='date')
    df_unemp_state = df_unemp_state.sort_values(by='Date')
    df_unemp_national = df_unemp_national.sort_values(by='DATE')
    df_cpi = df_cpi.sort_values(by='DATE')
    df_orig = df_orig.sort_values(by='first_payment_date')
    
    hpi_tolerance = pd.Timedelta(days=90)
    other_tolerance = pd.Timedelta(days=30) 

    df_orig = pd.merge_asof(
        df_orig,
        df_hpi[['state', 'date', 'index_sa_state_mom12']],
        left_on='first_payment_date',
        right_on='date',
        left_by='property_state',
        right_by='state',
        tolerance=hpi_tolerance,
        direction='backward'
    )
    df_orig.drop(['state', 'date'], axis=1, inplace=True)

    df_orig = pd.merge_asof(
        df_orig,
        df_unemp_state[['State Abbreviation', 'Date', 'Unemployment Rate']],
        left_on='first_payment_date',
        right_on='Date',
        left_by='property_state',
        right_by='State Abbreviation',
        tolerance=other_tolerance,
        direction='backward'
    )
    df_orig.rename(columns={'Unemployment Rate': 'State Unemployment Rate'}, inplace=True)
    df_orig.drop(['State Abbreviation', 'Date'], axis=1, inplace=True)

    df_orig = pd.merge_asof(
        df_orig,
        df_unemp_national[['DATE', 'UNRATE']],
        left_on='first_payment_date',
        right_on='DATE',
        tolerance=other_tolerance,
        direction='backward'
    )
    df_orig.rename(columns={'UNRATE': 'National Unemployment Rate'}, inplace=True)
    df_orig.drop(['DATE'], axis=1, inplace=True)

    df_orig = pd.merge_asof(
        df_orig,
        df_cpi[['DATE', 'CPIAUCSL']],
        left_on='first_payment_date',
        right_on='DATE',
        tolerance=other_tolerance,
        direction='backward'
    )
    df_orig.drop(['DATE'], axis=1, inplace=True)

    df_orig.drop_duplicates(inplace=True)
    df_orig.reset_index(drop=True, inplace=True)

    return df_orig

def bin_columns(df, binning_config):
    for col, (bins, labels) in binning_config.items():
        bin_col_name = f"{col}_bins"
        df[bin_col_name] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df

def preprocess_data(X_train, X_test, X_val, numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    X_val_preprocessed = preprocessor.transform(X_val)

    feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([numerical_features, feature_names])
    
    return X_train_preprocessed, X_test_preprocessed, X_val_preprocessed, feature_names

def plot_density(y_pred_proba, y_pred, y_test):
    df_results = pd.DataFrame({'y_pred_proba': y_pred_proba, 'obs': y_test, 'y_pred': y_pred})
    df_results['group'] = pd.qcut(df_results['y_pred_proba'], q=20, labels=False, duplicates='drop')
    grouped = df_results.groupby('group').agg({'y_pred_proba': 'mean', 'obs': 'mean'})

    x = np.arange(len(grouped))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.bar(x - 0.2, grouped['obs'], 0.4, color='salmon', label='Mean observed value')
    ax1.bar(x + 0.2, grouped['y_pred_proba'], 0.4, color='skyblue', label='Mean predicted probability')
    ax1.set_title('Comparison of Mean Predicted Probability and Mean Observed Value', fontsize=16)
    ax1.set_xlabel('Group', fontsize=14)
    ax1.set_ylabel('Mean Value', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped.index, rotation=90)
    ax1.legend()

    y_pred_proba_1 = y_pred_proba[y_test == 1]
    y_pred_proba_0 = y_pred_proba[y_test == 0]

    sns.set_theme(style='whitegrid', font_scale=1.2)

    sns.kdeplot(y_pred_proba_0, color='dodgerblue', linewidth=2, label='y == 0', fill=True, alpha=0.5, ax=ax2)
    sns.kdeplot(y_pred_proba_1, color='tomato', linewidth=2, label='y == 1', fill=True, alpha=0.5, ax=ax2)

    ax2.set_xlabel('Predicted Probability', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title('Density Plot of Predicted Probabilities', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper center')

    sns.despine(left=True, bottom=True, ax=ax2)
    ax2.tick_params(axis='both', labelsize=12)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

def plot_optimization_history(study):
        vis.plot_optimization_history(study).show()
        vis.plot_param_importances(study).show()
        vis.plot_slice(study).show()


def compute_feature_importance(feature, model, baseline_auc, X_train, X_test, X_val, y_val, n, numerical_features_ml, categorical_features_ml):
    importance_scores = []
    for _ in range(n):
        X_val_permuted = X_val.copy()
        X_val_permuted[feature] = np.random.permutation(X_val_permuted[feature])
        
        _, _, X_val_permuted_preprocessed, _ = preprocess_data(
            X_train, X_test, X_val_permuted, numerical_features_ml, categorical_features_ml
        )
        
        permuted_auc = roc_auc_score(y_val, model.predict_proba(X_val_permuted_preprocessed)[:, 1])
        importance_scores.append(baseline_auc - permuted_auc)
    
    return feature, importance_scores

def compute_group_permutation_importance(model, X_train, X_test, X_val_preprocessed, X_val, y_val, columns, n=5):
    baseline_auc = roc_auc_score(y_val, model.predict_proba(X_val_preprocessed)[:, 1])
    
    args = [
        (feature, model, baseline_auc, X_train, X_test, X_val, y_val, n, numerical_features_ml, categorical_features_ml)
        for feature in columns
    ]
    
    with Pool(cpu_count() - 1) as pool:
        results = pool.starmap(compute_feature_importance, args)
    
    importance_dict = {feature: scores for feature, scores in results}
    
    importance_df = pd.DataFrame([
        {'Feature': feature, 'Importance Mean': np.mean(scores), 'Importance Std': np.std(scores), 'Scores': scores}
        for feature, scores in importance_dict.items()
    ])

    importance_df['Importance Mean'] = importance_df['Importance Mean'] / np.sum(importance_df['Importance Mean'])
    
    return importance_df.sort_values(by='Importance Mean', ascending=False)

def plot_feature_importance(importance_df, model):
    title=f'{model.model_name} Feature Importance -- Permutation Importance'
    plt.figure(figsize=(10, 6))
    
    features = importance_df['Feature']
    scores = importance_df['Scores']
    
    plt.boxplot(scores, vert=False, labels=features, patch_artist=True)
    
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title, weight='bold')
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_tree_feature_importance(model, feature_names):
    title=f'{model.model_name} Feature Importance -- Tree-based (impurity-based)' 
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(30)
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title, weight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_model_feature_importance_comparison(model_importance_dfs, model_names):
    title = "Feature Importance Comparison Across Models"

    merged_df = model_importance_dfs[0][['Importance Mean', 'Feature']].rename(columns={'Importance Mean': model_names[0]})
    for i in range(1, len(model_importance_dfs)):
        merged_df = merged_df.merge(
            model_importance_dfs[i][['Importance Mean', 'Feature']].rename(columns={'Importance Mean': model_names[i]}),
            on='Feature',
            how='outer'  
        )

    for model_name in model_names:
        merged_df[model_name] = merged_df[model_name]

    merged_df.fillna(0, inplace=True)

    merged_df.set_index('Feature', inplace=True)
    merged_df['Average Importance'] = merged_df[model_names].mean(axis=1)
    merged_df.sort_values(by='Average Importance', ascending=False, inplace=True)

    plt.figure(figsize=(14, 8))
    plt.imshow(merged_df, aspect='auto', cmap='Blues', interpolation='none')
    plt.colorbar(label='Normalized Importance', orientation='vertical')

    for i in range(len(merged_df.index)):
        for j in range(len(merged_df.columns)):
            value = merged_df.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=8, color='black')

    plt.xticks(ticks=np.arange(len(merged_df.columns)), labels=merged_df.columns, rotation=15, fontsize=10)
    plt.yticks(ticks=np.arange(len(merged_df.index)), labels=merged_df.index, fontsize=10)
    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.tight_layout()
    plt.grid(False)
    plt.show()
