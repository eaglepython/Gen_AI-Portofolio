# Model Documentation

## Overview
This document describes the development, validation, and deployment of credit risk models (PD, LGD, EAD) and CECL expected loss calculations for CIB portfolios.

## Model List
- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD)
- CECL Expected Loss

## Regulatory Alignment
- SR 11-7, OCC 2011-12, Basel III, CECL

## Data Sources
- CIB loan and borrower data
- Financial statements
- Macroeconomic scenarios

## Methodology
- Data preprocessing, feature engineering
- Model development (logistic regression, survival analysis, XGBoost)
- Validation and benchmarking
- Scenario analysis for CECL

## Deployment
- REST API, batch scoring, dashboard
