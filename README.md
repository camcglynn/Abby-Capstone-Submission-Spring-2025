# Abby Chatbot

A specialized chatbot designed to provide accurate and supportive information about reproductive healthcare.

## Project Structure

```
.
├── app.py                 # Main FastAPI application
├── chatbot/              # Core chatbot implementation
├── config/               # Configuration files
├── data/                # Data files (managed separately)
├── deployment/          # Deployment configuration
│   ├── cloudformation/  # AWS CloudFormation templates
│   ├── docs/           # Deployment documentation
│   ├── scripts/        # Deployment scripts
├── models/             # ML model implementations
├── nltk_data/         # NLTK data files (managed separately)
├── scripts/           # Utility scripts
├── static/           # Static web assets
├── templates/        # HTML templates
├── terraform/        # Terraform configuration
└── utils/           # Utility functions
```

## Prerequisites

- Python 3.8+
- pip
- virtualenv or conda
- AWS CLI (for deployment)

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sahanasankar13/Abby_Chatbot.git
   cd Abby-Capstone-Submission-Spring-2025/Abby_3.0/
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the application:
   ```bash
   python app.py
   ```

The application will be available at http://localhost:#### (your local server #; ex: 8000)

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Data and Model Management

The following directories are managed separately and not included in version control:
- `data/`: Contains training data and configuration files
- `nltk_data/`: Contains NLTK data files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

For questions or support, please contact chloeamcglynn@gmail.com
