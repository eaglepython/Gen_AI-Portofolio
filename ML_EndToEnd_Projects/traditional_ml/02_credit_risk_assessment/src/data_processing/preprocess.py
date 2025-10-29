"""
Credit Risk Assessment - Data Processing Module
Handles loading, cleaning, and preprocessing of credit data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDataProcessor:
    """Process credit risk data for machine learning models."""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.preprocessors = {}
        self.feature_names = []
        self.target_column = 'default_risk'
        
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'data_dir': 'data',
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'handle_imbalance': True,
            'feature_selection': True,
            'outlier_threshold': 3.0
        }
    
    def load_data(self, data_source: str = "synthetic") -> pd.DataFrame:
        """Load credit data from various sources."""
        logger.info(f"Loading data from source: {data_source}")
        
        if data_source == "synthetic":
            return self._generate_synthetic_credit_data()
        elif data_source == "german_credit":
            return self._load_german_credit_data()
        elif data_source == "lending_club":
            return self._load_lending_club_data()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def _generate_synthetic_credit_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic credit data for demonstration."""
        logger.info(f"Generating {n_samples} synthetic credit records...")
        
        np.random.seed(42)
        
        # Customer demographics
        age = np.random.normal(40, 12, n_samples).clip(18, 80).astype(int)
        income = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000)
        
        # Employment information
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)
        job_categories = ['Professional', 'Manager', 'Skilled', 'Service', 'Other']
        job_category = np.random.choice(job_categories, n_samples, p=[0.3, 0.2, 0.25, 0.15, 0.1])
        
        # Financial information
        credit_amount = np.random.lognormal(9, 1, n_samples).clip(1000, 100000)
        existing_credits = np.random.poisson(2, n_samples).clip(0, 10)
        
        # Credit history
        credit_history_length = np.random.exponential(8, n_samples).clip(0, 30)
        previous_defaults = np.random.binomial(3, 0.1, n_samples)
        
        # Loan characteristics
        loan_purpose = np.random.choice(['Auto', 'Home', 'Personal', 'Business', 'Education'], 
                                       n_samples, p=[0.25, 0.3, 0.25, 0.15, 0.05])
        loan_duration = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.25, 0.35, 0.25, 0.05])
        
        # Payment behavior
        debt_to_income = (credit_amount * 0.1) / (income / 12)  # Monthly payment ratio
        debt_to_income = debt_to_income.clip(0, 1)
        
        # Property and collateral
        property_type = np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], 
                                        n_samples, p=[0.4, 0.3, 0.25, 0.05])
        property_value = np.where(property_type.isin(['Own', 'Mortgage']), 
                                 np.random.lognormal(12, 0.5, n_samples), 0)
        
        # Guarantors and co-applicants
        has_guarantor = np.random.binomial(1, 0.15, n_samples)
        has_co_applicant = np.random.binomial(1, 0.1, n_samples)
        
        # Banking relationship
        bank_account_type = np.random.choice(['Checking', 'Savings', 'Both', 'None'], 
                                           n_samples, p=[0.4, 0.2, 0.35, 0.05])
        account_balance = np.random.lognormal(8, 1.5, n_samples).clip(0, 50000)
        
        # Risk factors
        risk_score = (
            (age < 25) * 0.2 +
            (income < 30000) * 0.3 +
            (employment_length < 1) * 0.25 +
            (debt_to_income > 0.4) * 0.4 +
            (previous_defaults > 0) * 0.5 +
            (existing_credits > 3) * 0.2 +
            (property_type == 'Other') * 0.15 +
            (bank_account_type == 'None') * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Generate target variable (default risk)
        default_probability = 1 / (1 + np.exp(-3 * (risk_score - 0.5)))  # Sigmoid
        default_risk = np.random.binomial(1, default_probability, n_samples)
        
        # Create DataFrame
        credit_data = pd.DataFrame({
            # Demographics
            'age': age,
            'income': income,
            'employment_length': employment_length,
            'job_category': job_category,
            
            # Loan information
            'credit_amount': credit_amount,
            'loan_purpose': loan_purpose,
            'loan_duration': loan_duration,
            
            # Credit history
            'credit_history_length': credit_history_length,
            'existing_credits': existing_credits,
            'previous_defaults': previous_defaults,
            
            # Financial ratios
            'debt_to_income_ratio': debt_to_income,
            
            # Property and collateral
            'property_type': property_type,
            'property_value': property_value,
            'has_guarantor': has_guarantor,
            'has_co_applicant': has_co_applicant,
            
            # Banking
            'bank_account_type': bank_account_type,
            'account_balance': account_balance,
            
            # Target variable
            'default_risk': default_risk
        })
        
        # Add some missing values to make it realistic
        missing_cols = ['employment_length', 'property_value', 'account_balance']
        for col in missing_cols:
            if col in credit_data.columns:
                missing_idx = np.random.choice(
                    credit_data.index, 
                    size=int(0.05 * len(credit_data)), 
                    replace=False
                )
                credit_data.loc[missing_idx, col] = np.nan
        
        logger.info(f"Generated credit data shape: {credit_data.shape}")
        logger.info(f"Default rate: {credit_data['default_risk'].mean():.2%}")
        
        return credit_data
    
    def _load_german_credit_data(self) -> pd.DataFrame:
        """Load German Credit Dataset."""
        logger.info("Loading German Credit Dataset...")
        
        # This would typically load from UCI ML repository
        # For now, generate similar synthetic data
        return self._generate_synthetic_credit_data(n_samples=1000)
    
    def _load_lending_club_data(self) -> pd.DataFrame:
        """Load Lending Club dataset."""
        logger.info("Loading Lending Club dataset...")
        
        # This would load actual Lending Club data
        # For now, generate synthetic data with similar characteristics
        return self._generate_synthetic_credit_data(n_samples=50000)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data."""
        logger.info("Cleaning credit data...")
        
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_size - len(df)} duplicate records")
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        # Numerical columns - use median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column]
        
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value:.2f}")
        
        # Categorical columns - use mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
        
        # Data validation
        logger.info("Validating data...")
        
        # Age should be reasonable
        df = df[(df['age'] >= 18) & (df['age'] <= 100)]
        
        # Income should be positive
        df = df[df['income'] > 0]
        
        # Credit amount should be positive
        df = df[df['credit_amount'] > 0]
        
        # Remove outliers using IQR method
        df = self._remove_outliers(df, numerical_cols)
        
        logger.info(f"Final cleaned data shape: {df.shape}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        logger.info("Removing outliers...")
        
        initial_size = len(df)
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Remove extreme outliers (beyond 3 IQR)
                    extreme_lower = Q1 - 3 * IQR
                    extreme_upper = Q3 + 3 * IQR
                    extreme_outliers = (df[col] < extreme_lower) | (df[col] > extreme_upper)
                    
                    df = df[~extreme_outliers]
                    logger.info(f"Removed {extreme_outliers.sum()} extreme outliers from {col}")
        
        final_size = len(df)
        logger.info(f"Removed {initial_size - final_size} total outliers")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features for better model performance."""
        logger.info("Engineering features...")
        
        # Financial ratios
        df['income_to_credit_ratio'] = df['income'] / df['credit_amount']
        df['monthly_payment'] = df['credit_amount'] / df['loan_duration']
        df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / df['income']
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 50, 65, 100], 
                                labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior', 'Elderly'])
        
        # Income groups
        df['income_group'] = pd.cut(df['income'], 
                                   bins=[0, 30000, 50000, 75000, 100000, float('inf')], 
                                   labels=['Low', 'Lower_Middle', 'Middle', 'Upper_Middle', 'High'])
        
        # Credit utilization
        df['credit_utilization'] = df['credit_amount'] / (df['income'] * 0.1)  # Assuming 10% of income
        df['credit_utilization'] = df['credit_utilization'].clip(0, 5)  # Cap at 500%
        
        # Experience factors
        df['employment_stability'] = np.where(df['employment_length'] >= 5, 'Stable', 'Unstable')
        df['credit_experience'] = np.where(df['credit_history_length'] >= 3, 'Experienced', 'Novice')
        
        # Risk indicators
        df['high_risk_job'] = df['job_category'].isin(['Service', 'Other']).astype(int)
        df['multiple_credits'] = (df['existing_credits'] > 2).astype(int)
        df['has_defaults'] = (df['previous_defaults'] > 0).astype(int)
        
        # Property indicators
        df['owns_property'] = df['property_type'].isin(['Own', 'Mortgage']).astype(int)
        df['property_to_credit_ratio'] = np.where(df['property_value'] > 0, 
                                                 df['property_value'] / df['credit_amount'], 0)
        
        # Banking relationship
        df['good_banking_relationship'] = df['bank_account_type'].isin(['Checking', 'Both']).astype(int)
        df['account_balance_to_income'] = df['account_balance'] / df['income']
        
        # Interaction features
        df['young_high_credit'] = ((df['age'] < 30) & (df['credit_amount'] > df['credit_amount'].median())).astype(int)
        df['low_income_high_debt'] = ((df['income'] < 40000) & (df['debt_to_income_ratio'] > 0.3)).astype(int)
        
        logger.info(f"Added engineered features. New shape: {df.shape}")
        
        return df
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # One-hot encode categorical variables
        if categorical_cols:
            logger.info(f"One-hot encoding: {categorical_cols}")
            
            # Use pd.get_dummies for simplicity
            X_encoded = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
            
            logger.info(f"Encoded features shape: {X_encoded.shape}")
        else:
            X_encoded = X
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        # Combine back with target
        result_df = pd.concat([X_encoded, y], axis=1)
        
        return result_df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling numerical features...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Identify numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            # Initialize scaler
            scaler = StandardScaler()
            
            # Fit and transform numerical columns
            X_scaled = X.copy()
            X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            
            # Store scaler for later use
            self.preprocessors['scaler'] = scaler
            
            logger.info(f"Scaled {len(numerical_cols)} numerical features")
        else:
            X_scaled = X
        
        # Combine back with target
        result_df = pd.concat([X_scaled, y], axis=1)
        
        return result_df
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = self.config['validation_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['random_state'],
            stratify=y_temp
        )
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        # Check class distribution
        for split_name, y_split in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
            default_rate = y_split.mean()
            logger.info(f"  {split_name} default rate: {default_rate:.2%}")
        
        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE or other techniques."""
        if not self.config.get('handle_imbalance', False):
            return X_train, y_train
        
        logger.info("Handling class imbalance...")
        
        try:
            from imblearn.over_sampling import SMOTE
            
            # Apply SMOTE
            smote = SMOTE(random_state=self.config['random_state'])
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"Original distribution: {y_train.value_counts().to_dict()}")
            logger.info(f"Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        
        except ImportError:
            logger.warning("imblearn not available. Skipping class imbalance handling.")
            return X_train, y_train
    
    def save_processed_data(self, data_splits: Dict, output_dir: str):
        """Save processed data."""
        logger.info(f"Saving processed data to {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, (X, y) in data_splits.items():
            # Combine features and target
            combined_df = pd.concat([X, y], axis=1)
            
            # Save to CSV
            file_path = output_path / f"{split_name}.csv"
            combined_df.to_csv(file_path, index=False)
            logger.info(f"Saved {split_name} data: {len(combined_df)} records")
        
        # Save feature names
        feature_names_path = output_path / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        # Save preprocessors
        import joblib
        if self.preprocessors:
            preprocessors_path = output_path / "preprocessors.pkl"
            joblib.dump(self.preprocessors, preprocessors_path)
            logger.info("Saved preprocessors")
    
    def process_all(self, data_source: str = "synthetic", output_dir: str = "data/processed") -> Dict:
        """Run the complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline...")
        
        # Load data
        raw_data = self.load_data(data_source)
        
        # Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Engineer features
        featured_data = self.engineer_features(cleaned_data)
        
        # Encode categorical features
        encoded_data = self.encode_features(featured_data)
        
        # Scale features
        scaled_data = self.scale_features(encoded_data)
        
        # Split data
        data_splits = self.split_data(scaled_data)
        
        # Handle class imbalance for training set
        X_train, y_train = data_splits['train']
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        data_splits['train'] = (X_train_balanced, y_train_balanced)
        
        # Save processed data
        self.save_processed_data(data_splits, output_dir)
        
        logger.info("Data processing pipeline completed successfully")
        
        return data_splits


def main():
    """Main function to run data processing."""
    processor = CreditDataProcessor()
    
    # Process data
    data_splits = processor.process_all(
        data_source="synthetic",
        output_dir="data/processed"
    )
    
    print("\n=== Credit Data Processing Summary ===")
    for split_name, (X, y) in data_splits.items():
        print(f"{split_name}: {len(X)} samples, {len(X.columns)} features")
        print(f"  Default rate: {y.mean():.2%}")
    
    print(f"\nFeatures ({len(processor.feature_names)}):")
    for i, feature in enumerate(processor.feature_names[:10]):  # Show first 10
        print(f"  {i+1}. {feature}")
    if len(processor.feature_names) > 10:
        print(f"  ... and {len(processor.feature_names) - 10} more")


if __name__ == "__main__":
    main()