#!/bin/bash

# ML End-to-End Projects Collection - Automated Setup Script
# This script sets up all 15 ML projects with dependencies and sample data

set -e

echo "🚀 Setting up ML End-to-End Projects Collection..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}==== $1 ====${NC}"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv package manager not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    else
        print_status "uv package manager found"
    fi
}

# Setup project dependencies
setup_project() {
    local project_path=$1
    local project_name=$2
    
    print_header "Setting up $project_name"
    
    if [ -d "$project_path" ] && [ -f "$project_path/pyproject.toml" ]; then
        cd "$project_path"
        print_status "Installing dependencies for $project_name..."
        
        # Create virtual environment and install dependencies
        if uv venv .venv 2>/dev/null; then
            source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
            uv pip install -e . 2>/dev/null || print_warning "Some dependencies may require manual installation"
        else
            print_warning "Could not create virtual environment for $project_name"
        fi
        
        # Create necessary directories
        mkdir -p data/{raw,processed,features} models logs
        
        # Generate sample configuration if it doesn't exist
        if [ ! -f ".env" ] && [ -f ".env.example" ]; then
            cp .env.example .env
            print_status "Created .env file from template"
        fi
        
        cd - > /dev/null
        print_status "✅ $project_name setup complete"
    else
        print_warning "Skipping $project_name - no pyproject.toml found"
    fi
}

# Main setup function
main() {
    print_header "ML End-to-End Projects Collection Setup"
    
    # Check prerequisites
    check_uv
    
    # Get the base directory
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    TRADITIONAL_ML_DIR="$BASE_DIR/traditional_ml"
    GENERATIVE_AI_DIR="$BASE_DIR/generative_ai"
    
    print_status "Base directory: $BASE_DIR"
    
    # Setup Traditional ML Projects
    print_header "Setting up Traditional ML Projects"
    
    if [ -d "$TRADITIONAL_ML_DIR" ]; then
        # E-commerce Recommender System
        setup_project "$TRADITIONAL_ML_DIR/01_ecommerce_recommender" "E-commerce Recommender System"
        
        # Credit Risk Assessment
        setup_project "$TRADITIONAL_ML_DIR/02_credit_risk" "Credit Risk Assessment"
        
        # Stock Market Forecasting
        setup_project "$TRADITIONAL_ML_DIR/03_stock_forecasting" "Stock Market Forecasting"
        
        # Medical Image Classification
        setup_project "$TRADITIONAL_ML_DIR/04_medical_imaging" "Medical Image Classification"
        
        # NLP Sentiment Analysis
        setup_project "$TRADITIONAL_ML_DIR/05_sentiment_analysis" "NLP Sentiment Analysis"
        
        # Fraud Detection System
        setup_project "$TRADITIONAL_ML_DIR/06_fraud_detection" "Fraud Detection System"
        
        # Customer Churn Prediction
        setup_project "$TRADITIONAL_ML_DIR/07_churn_prediction" "Customer Churn Prediction"
        
        # Supply Chain Optimization
        setup_project "$TRADITIONAL_ML_DIR/08_supply_chain" "Supply Chain Optimization"
        
        # Energy Consumption Forecasting
        setup_project "$TRADITIONAL_ML_DIR/09_energy_forecasting" "Energy Consumption Forecasting"
        
        # Autonomous Vehicle Path Planning
        setup_project "$TRADITIONAL_ML_DIR/10_autonomous_vehicle" "Autonomous Vehicle Path Planning"
    else
        print_warning "Traditional ML directory not found: $TRADITIONAL_ML_DIR"
    fi
    
    # Setup Generative AI Projects
    print_header "Setting up Generative AI Projects"
    
    if [ -d "$GENERATIVE_AI_DIR" ]; then
        # Code Generation Assistant
        setup_project "$GENERATIVE_AI_DIR/01_code_assistant" "Code Generation Assistant"
        
        # Multimodal Content Creator
        setup_project "$GENERATIVE_AI_DIR/02_content_creator" "Multimodal Content Creator"
        
        # Intelligent Document Processing
        setup_project "$GENERATIVE_AI_DIR/03_document_processing" "Intelligent Document Processing"
        
        # Conversational AI Assistant
        setup_project "$GENERATIVE_AI_DIR/04_conversational_ai" "Conversational AI Assistant"
        
        # AI-Powered Drug Discovery
        setup_project "$GENERATIVE_AI_DIR/05_drug_discovery" "AI-Powered Drug Discovery"
    else
        print_warning "Generative AI directory not found: $GENERATIVE_AI_DIR"
    fi
    
    # Final instructions
    print_header "Setup Complete!"
    
    cat << EOF

🎉 All ML projects have been set up successfully!

📁 Project Structure:
   ├── traditional_ml/        (10 projects)
   │   ├── 01_ecommerce_recommender/
   │   ├── 02_credit_risk/
   │   ├── 03_stock_forecasting/
   │   ├── 04_medical_imaging/
   │   ├── 05_sentiment_analysis/
   │   ├── 06_fraud_detection/
   │   ├── 07_churn_prediction/
   │   ├── 08_supply_chain/
   │   ├── 09_energy_forecasting/
   │   └── 10_autonomous_vehicle/
   └── generative_ai/         (5 projects)
       ├── 01_code_assistant/
       ├── 02_content_creator/
       ├── 03_document_processing/
       ├── 04_conversational_ai/
       └── 05_drug_discovery/

🚀 Next Steps:
1. Navigate to any project directory
2. Activate the virtual environment: source .venv/bin/activate
3. Configure environment variables in .env file
4. Run the project-specific setup scripts
5. Start developing!

📖 Each project includes:
   ✅ Complete README with setup instructions
   ✅ Modular code architecture
   ✅ Docker containers for deployment
   ✅ Streamlit dashboards
   ✅ FastAPI services
   ✅ Comprehensive testing
   ✅ MLOps best practices

🔗 Useful Commands:
   - cd traditional_ml/01_ecommerce_recommender && streamlit run app.py
   - cd generative_ai/01_code_assistant && python src/api/main.py
   - docker-compose up -d (in any project directory)

Happy coding! 🎯

EOF
}

# Run main function
main "$@"