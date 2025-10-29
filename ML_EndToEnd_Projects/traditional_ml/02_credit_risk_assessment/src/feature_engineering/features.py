"""
Credit Risk Assessment - Feature Engineering Module
Advanced feature engineering for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CreditFeatureEngineer:
    """Advanced feature engineering for credit risk assessment."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_importance_scores = {}
        self.selected_features = []
        self.feature_transformers = {}
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for credit risk modeling."""
        logger.info("Creating advanced credit features...")
        
        # Make a copy to avoid modifying original
        df_featured = df.copy()
        
        # Financial stability indicators
        df_featured = self._create_stability_features(df_featured)
        
        # Risk aggregation features
        df_featured = self._create_risk_aggregation_features(df_featured)
        
        # Behavioral patterns
        df_featured = self._create_behavioral_features(df_featured)
        
        # Economic indicators
        df_featured = self._create_economic_features(df_featured)
        
        # Interaction features
        df_featured = self._create_interaction_features(df_featured)
        
        # Statistical features
        df_featured = self._create_statistical_features(df_featured)
        
        logger.info(f"Created advanced features. Shape: {df_featured.shape}")
        
        return df_featured
    
    def _create_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial stability indicators."""
        logger.info("Creating financial stability features...")
        
        # Employment stability score
        df['employment_stability_score'] = (
            (df['employment_length'] >= 2).astype(int) * 0.3 +
            (df['employment_length'] >= 5).astype(int) * 0.3 +
            (df['employment_length'] >= 10).astype(int) * 0.4
        )
        
        # Income stability (based on income consistency)
        df['income_stability'] = np.where(
            df['income'] > df['income'].quantile(0.75), 'High',
            np.where(df['income'] > df['income'].quantile(0.25), 'Medium', 'Low')
        )
        
        # Credit history stability
        df['credit_stability_score'] = (
            (df['credit_history_length'] >= 1).astype(int) * 0.2 +
            (df['credit_history_length'] >= 3).astype(int) * 0.3 +
            (df['credit_history_length'] >= 5).astype(int) * 0.3 +
            (df['previous_defaults'] == 0).astype(int) * 0.2
        )
        
        # Overall stability score
        df['overall_stability_score'] = (
            df['employment_stability_score'] * 0.4 +
            df['credit_stability_score'] * 0.6
        )
        
        # Debt management capability
        df['debt_management_score'] = np.where(
            df['debt_to_income_ratio'] <= 0.2, 1.0,
            np.where(df['debt_to_income_ratio'] <= 0.4, 0.7,
                    np.where(df['debt_to_income_ratio'] <= 0.6, 0.4, 0.1))
        )
        
        return df
    
    def _create_risk_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk aggregation features."""
        logger.info("Creating risk aggregation features...")
        
        # Age-based risk factors
        df['age_risk_factor'] = np.where(
            df['age'] < 25, 0.8,  # Higher risk for young
            np.where(df['age'] > 65, 0.7,  # Higher risk for elderly
                    np.where(df['age'].between(35, 50), 0.3, 0.5))  # Lower risk for middle-aged
        )
        
        # Income-based risk factors
        income_percentiles = df['income'].quantile([0.1, 0.25, 0.75, 0.9])
        df['income_risk_factor'] = np.where(
            df['income'] <= income_percentiles[0.1], 0.9,  # Very low income
            np.where(df['income'] <= income_percentiles[0.25], 0.7,  # Low income
                    np.where(df['income'] >= income_percentiles[0.9], 0.2,  # Very high income
                            np.where(df['income'] >= income_percentiles[0.75], 0.3, 0.5)))  # High income
        )
        
        # Combined risk score
        df['combined_risk_score'] = (
            df['age_risk_factor'] * 0.2 +
            df['income_risk_factor'] * 0.3 +
            (df['debt_to_income_ratio'] > 0.5).astype(int) * 0.25 +
            (df['previous_defaults'] > 0).astype(int) * 0.25
        )
        
        # Credit utilization risk
        df['credit_utilization_risk'] = np.where(
            df['debt_to_income_ratio'] > 0.8, 'High',
            np.where(df['debt_to_income_ratio'] > 0.4, 'Medium', 'Low')
        )
        
        # Multiple credit risk
        df['multiple_credit_risk'] = np.where(
            df['existing_credits'] > 4, 'High',
            np.where(df['existing_credits'] > 2, 'Medium', 'Low')
        )
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features."""
        logger.info("Creating behavioral features...")
        
        # Credit seeking behavior
        df['credit_seeking_intensity'] = df['existing_credits'] / (df['credit_history_length'] + 1)
        
        # Risk taking propensity
        df['risk_taking_propensity'] = (
            (df['loan_purpose'] == 'Business').astype(int) * 0.4 +
            (df['credit_amount'] > df['income']).astype(int) * 0.3 +
            (df['has_guarantor'] == 0).astype(int) * 0.3
        )
        
        # Financial planning behavior
        df['financial_planning_score'] = (
            (df['has_guarantor']).astype(int) * 0.2 +
            (df['has_co_applicant']).astype(int) * 0.2 +
            (df['property_value'] > 0).astype(int) * 0.3 +
            (df['account_balance'] > df['income'] * 0.1).astype(int) * 0.3
        )
        
        # Loan shopping behavior (inferred)
        df['loan_shopping_behavior'] = np.where(
            (df['loan_purpose'] == 'Personal') & (df['credit_amount'] > df['income'] * 0.5),
            'Aggressive',
            np.where((df['has_guarantor']) | (df['has_co_applicant']), 'Conservative', 'Moderate')
        )
        
        # Payment capacity
        monthly_income = df['income'] / 12
        estimated_monthly_payment = df['credit_amount'] / df['loan_duration']
        df['payment_capacity_ratio'] = estimated_monthly_payment / monthly_income
        
        df['payment_capacity_category'] = np.where(
            df['payment_capacity_ratio'] > 0.5, 'Strained',
            np.where(df['payment_capacity_ratio'] > 0.3, 'Tight', 'Comfortable')
        )
        
        return df
    
    def _create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic context features."""
        logger.info("Creating economic features...")
        
        # Loan amount relative to market
        df['credit_amount_percentile'] = df['credit_amount'].rank(pct=True)
        df['large_loan_indicator'] = (df['credit_amount_percentile'] > 0.9).astype(int)
        
        # Income relative to peers
        df['income_percentile'] = df['income'].rank(pct=True)
        df['high_income_indicator'] = (df['income_percentile'] > 0.8).astype(int)
        
        # Economic efficiency ratios
        df['income_per_year_of_age'] = df['income'] / df['age']
        df['credit_per_year_of_history'] = df['credit_amount'] / (df['credit_history_length'] + 1)
        
        # Asset-to-debt ratios
        total_assets = df['property_value'] + df['account_balance']
        df['asset_to_debt_ratio'] = total_assets / (df['credit_amount'] + 1)
        
        # Liquidity indicators
        df['liquidity_ratio'] = df['account_balance'] / (df['income'] / 12)  # Months of expenses covered
        df['liquidity_category'] = np.where(
            df['liquidity_ratio'] > 6, 'High',
            np.where(df['liquidity_ratio'] > 3, 'Medium', 'Low')
        )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        logger.info("Creating interaction features...")
        
        # Age-Income interactions
        df['young_low_income'] = ((df['age'] < 30) & (df['income'] < 40000)).astype(int)
        df['mature_high_income'] = ((df['age'].between(40, 55)) & (df['income'] > 75000)).astype(int)
        
        # Employment-Credit interactions
        df['stable_employment_high_credit'] = (
            (df['employment_length'] >= 5) & (df['credit_amount'] > df['income'])
        ).astype(int)
        
        # Property-Credit interactions
        df['property_owner_large_loan'] = (
            (df['property_value'] > 0) & (df['credit_amount'] > 50000)
        ).astype(int)
        
        # Risk combinations
        df['multiple_risk_factors'] = (
            (df['age'] < 25).astype(int) +
            (df['employment_length'] < 2).astype(int) +
            (df['previous_defaults'] > 0).astype(int) +
            (df['debt_to_income_ratio'] > 0.5).astype(int) +
            (df['existing_credits'] > 3).astype(int)
        )
        
        # Positive combinations
        df['positive_factors'] = (
            (df['property_value'] > 0).astype(int) +
            (df['has_guarantor']).astype(int) +
            (df['employment_length'] >= 5).astype(int) +
            (df['income'] > 60000).astype(int) +
            (df['account_balance'] > 10000).astype(int)
        )
        
        # Loan purpose and amount interactions
        df['business_large_loan'] = (
            (df['loan_purpose'] == 'Business') & (df['credit_amount'] > df['income'])
        ).astype(int)
        
        df['home_loan_property_owner'] = (
            (df['loan_purpose'] == 'Home') & (df['property_value'] > 0)
        ).astype(int)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        logger.info("Creating statistical features...")
        
        # Z-scores for key numerical variables
        numerical_cols = ['age', 'income', 'credit_amount', 'employment_length', 'debt_to_income_ratio']
        
        for col in numerical_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean_val) / (std_val + 1e-8)
                
                # Flag outliers
                df[f'{col}_outlier'] = (np.abs(df[f'{col}_zscore']) > 2).astype(int)
        
        # Percentile features
        for col in ['income', 'credit_amount']:
            if col in df.columns:
                df[f'{col}_percentile_bin'] = pd.qcut(df[col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # Ratios and relative measures
        df['credit_to_income_percentile'] = (df['credit_amount'] / df['income']).rank(pct=True)
        df['age_employment_ratio'] = df['employment_length'] / df['age']
        
        # Moving averages and trends (simulated)
        # In real scenario, these would be based on historical data
        df['income_trend'] = np.random.choice(['Increasing', 'Stable', 'Decreasing'], 
                                             size=len(df), p=[0.3, 0.5, 0.2])
        df['credit_usage_trend'] = np.random.choice(['Growing', 'Stable', 'Reducing'], 
                                                   size=len(df), p=[0.4, 0.4, 0.2])
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'random_forest') -> List[str]:
        """Select most important features."""
        logger.info(f"Selecting features using {method}...")
        
        if method == 'random_forest':
            return self._select_features_rf(X, y)
        elif method == 'statistical':
            return self._select_features_statistical(X, y)
        elif method == 'rfe':
            return self._select_features_rfe(X, y)
        elif method == 'combined':
            return self._select_features_combined(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _select_features_rf(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Random Forest importance."""
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_encoded, y)
        
        # Get feature importance
        feature_importance = pd.Series(rf.feature_importances_, index=X_encoded.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Select top features
        n_features = min(50, len(feature_importance))  # Top 50 or all if less
        top_features = feature_importance.head(n_features).index.tolist()
        
        self.feature_importance_scores['random_forest'] = feature_importance
        
        logger.info(f"Selected {len(top_features)} features using Random Forest")
        
        return top_features
    
    def _select_features_statistical(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using statistical tests."""
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=50)
        selector.fit(X_encoded, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X_encoded.columns[selected_mask].tolist()
        
        # Store scores
        feature_scores = pd.Series(selector.scores_, index=X_encoded.columns)
        self.feature_importance_scores['statistical'] = feature_scores.sort_values(ascending=False)
        
        logger.info(f"Selected {len(selected_features)} features using statistical tests")
        
        return selected_features
    
    def _select_features_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Use Random Forest as estimator for RFE
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Perform RFE
        n_features = min(50, X_encoded.shape[1])
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X_encoded, y)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = X_encoded.columns[selected_mask].tolist()
        
        # Store rankings
        feature_rankings = pd.Series(rfe.ranking_, index=X_encoded.columns)
        self.feature_importance_scores['rfe'] = feature_rankings.sort_values()
        
        logger.info(f"Selected {len(selected_features)} features using RFE")
        
        return selected_features
    
    def _select_features_combined(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combine multiple feature selection methods."""
        logger.info("Combining multiple feature selection methods...")
        
        # Get features from different methods
        rf_features = set(self._select_features_rf(X, y))
        stat_features = set(self._select_features_statistical(X, y))
        rfe_features = set(self._select_features_rfe(X, y))
        
        # Features selected by at least 2 methods
        consensus_features = []
        all_features = rf_features | stat_features | rfe_features
        
        for feature in all_features:
            count = (feature in rf_features) + (feature in stat_features) + (feature in rfe_features)
            if count >= 2:
                consensus_features.append(feature)
        
        # If not enough consensus features, add top features from each method
        if len(consensus_features) < 30:
            additional_features = list(rf_features | stat_features | rfe_features)
            for feature in additional_features:
                if feature not in consensus_features and len(consensus_features) < 50:
                    consensus_features.append(feature)
        
        logger.info(f"Selected {len(consensus_features)} features using combined method")
        
        return consensus_features
    
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, 
                                 selected_cols: List[str] = None) -> pd.DataFrame:
        """Create polynomial features for selected numerical columns."""
        logger.info(f"Creating polynomial features of degree {degree}...")
        
        X_poly = X.copy()
        
        # Select numerical columns
        if selected_cols is None:
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Select only most important numerical features to avoid explosion
            selected_cols = numerical_cols[:5]  # Limit to top 5 to avoid too many features
        
        for col in selected_cols:
            if col in X.columns:
                for d in range(2, degree + 1):
                    X_poly[f'{col}_poly_{d}'] = X[col] ** d
                
                # Square root and log transformations
                if X[col].min() >= 0:
                    X_poly[f'{col}_sqrt'] = np.sqrt(X[col])
                
                if X[col].min() > 0:
                    X_poly[f'{col}_log'] = np.log1p(X[col])
        
        logger.info(f"Created polynomial features. New shape: {X_poly.shape}")
        
        return X_poly
    
    def create_binned_features(self, X: pd.DataFrame, cols_to_bin: List[str] = None) -> pd.DataFrame:
        """Create binned versions of continuous variables."""
        logger.info("Creating binned features...")
        
        X_binned = X.copy()
        
        if cols_to_bin is None:
            cols_to_bin = ['age', 'income', 'credit_amount', 'employment_length']
        
        for col in cols_to_bin:
            if col in X.columns and X[col].dtype in [np.number]:
                try:
                    # Create equal-frequency bins
                    X_binned[f'{col}_bin'] = pd.qcut(X[col], q=5, duplicates='drop', 
                                                    labels=[f'{col}_low', f'{col}_med_low', 
                                                           f'{col}_medium', f'{col}_med_high', f'{col}_high'])
                    
                    # Create equal-width bins
                    X_binned[f'{col}_width_bin'] = pd.cut(X[col], bins=5, 
                                                         labels=[f'{col}_w1', f'{col}_w2', 
                                                                f'{col}_w3', f'{col}_w4', f'{col}_w5'])
                except Exception as e:
                    logger.warning(f"Could not create bins for {col}: {e}")
        
        logger.info(f"Created binned features. New shape: {X_binned.shape}")
        
        return X_binned
    
    def engineer_all_features(self, df: pd.DataFrame, target_column: str = 'default_risk') -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        logger.info("Running complete feature engineering pipeline...")
        
        # Separate features and target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            y = None
            X = df.copy()
        
        # Create advanced features
        X_featured = self.create_advanced_features(X)
        
        # Create polynomial features for selected columns
        key_numerical_cols = ['income', 'credit_amount', 'age', 'debt_to_income_ratio']
        key_numerical_cols = [col for col in key_numerical_cols if col in X_featured.columns]
        X_featured = self.create_polynomial_features(X_featured, degree=2, selected_cols=key_numerical_cols)
        
        # Create binned features
        X_featured = self.create_binned_features(X_featured)
        
        # Feature selection
        if y is not None:
            selected_features = self.select_features(X_featured, y, method='combined')
            
            # Keep only selected features
            # First encode categorical variables
            X_encoded = pd.get_dummies(X_featured, drop_first=True)
            
            # Select features that exist in encoded data
            existing_features = [f for f in selected_features if f in X_encoded.columns]
            X_final = X_encoded[existing_features]
            
            logger.info(f"Final feature set: {X_final.shape[1]} features")
            
            # Store selected features for later use
            self.selected_features = existing_features
            
            # Combine with target if provided
            if y is not None:
                result_df = pd.concat([X_final, y], axis=1)
            else:
                result_df = X_final
        else:
            # No target provided, return all engineered features
            result_df = pd.get_dummies(X_featured, drop_first=True)
        
        return result_df
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using stored feature engineering pipeline."""
        logger.info("Transforming new data using stored pipeline...")
        
        if not self.selected_features:
            logger.warning("No stored feature selection found. Using all features.")
            return pd.get_dummies(df, drop_first=True)
        
        # Apply same feature engineering
        X_featured = self.create_advanced_features(df)
        
        # Create polynomial features
        key_numerical_cols = ['income', 'credit_amount', 'age', 'debt_to_income_ratio']
        key_numerical_cols = [col for col in key_numerical_cols if col in X_featured.columns]
        X_featured = self.create_polynomial_features(X_featured, degree=2, selected_cols=key_numerical_cols)
        
        # Create binned features
        X_featured = self.create_binned_features(X_featured)
        
        # Encode categorical variables
        X_encoded = pd.get_dummies(X_featured, drop_first=True)
        
        # Select only the features that were selected during training
        missing_features = set(self.selected_features) - set(X_encoded.columns)
        extra_features = set(X_encoded.columns) - set(self.selected_features)
        
        if missing_features:
            logger.warning(f"Missing features in new data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X_encoded[feature] = 0
        
        if extra_features:
            logger.info(f"Dropping extra features: {len(extra_features)} features")
        
        # Select only the required features in the correct order
        X_final = X_encoded[self.selected_features]
        
        return X_final


def main():
    """Test feature engineering."""
    from .preprocess import CreditDataProcessor
    
    # Load and preprocess data
    processor = CreditDataProcessor()
    raw_data = processor.load_data("synthetic")
    cleaned_data = processor.clean_data(raw_data)
    
    # Initialize feature engineer
    engineer = CreditFeatureEngineer()
    
    # Engineer features
    featured_data = engineer.engineer_all_features(cleaned_data)
    
    print("\n=== Feature Engineering Summary ===")
    print(f"Original features: {cleaned_data.shape[1] - 1}")  # Exclude target
    print(f"Engineered features: {featured_data.shape[1] - 1}")  # Exclude target
    print(f"Feature improvement: {((featured_data.shape[1] - 1) / (cleaned_data.shape[1] - 1) - 1) * 100:.1f}%")
    
    if engineer.selected_features:
        print(f"\nSelected features ({len(engineer.selected_features)}):")
        for i, feature in enumerate(engineer.selected_features[:15]):  # Show first 15
            print(f"  {i+1}. {feature}")
        if len(engineer.selected_features) > 15:
            print(f"  ... and {len(engineer.selected_features) - 15} more")
    
    # Show feature importance if available
    if 'random_forest' in engineer.feature_importance_scores:
        print(f"\nTop 10 most important features:")
        top_features = engineer.feature_importance_scores['random_forest'].head(10)
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()