# Token Economics PoC

A proof-of-concept application that evaluates LLM token usage and costs across different cloud providers for loan evaluation tasks.

## Overview

This project provides a tool to compare token usage, costs, and performance across multiple cloud LLMs for a specific task. It uses LangChain to interface with multiple LLM providers, processes a transcript with a consistent prompt, and tracks token usage and costs.

The application includes:
- CLI-based batch runner (`driver.py`)
- Streamlit dashboard for visualizing results (`app.py`)
- Support for multiple LLM providers (OpenAI, GroqCloud)

## Models Supported

| Provider | Model ID | Description |
|----------|----------|-------------|
| OpenAI   | o3 (gpt-3.5-turbo-0125) | GPT-3.5 Turbo |
| OpenAI   | 4o (gpt-4o) | GPT-4o |
| OpenAI   | o4-mini (gpt-4o-mini) | GPT-4o Mini |
| GroqCloud | llama-3-70b-instruct | Llama 3 70B Instruct |
| GroqCloud | deepseek-r1 | DeepSeek R1 |

## Setup

### Prerequisites

- Python 3.11+
- API keys for OpenAI and GroqCloud

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd token-poc
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create your configuration file:
   ```
   cp config.yaml.example config.yaml
   ```

5. Edit `config.yaml` to add your API keys and adjust model settings if needed.

## Usage

### Running the Token Analyzer

To process the transcript with all configured models:

```
cd token-poc
python driver.py
```

This will:
1. Load the transcript and prompt
2. Process the transcript with each configured model
3. Collect token usage and cost data
4. Save the results to an Excel file in the `runs/` directory

### Viewing Results in the Dashboard

To launch the Streamlit dashboard:

```
cd token-poc
streamlit run app.py
```

The dashboard allows you to:
- View and compare token usage and costs across models
- Filter columns to display
- Visualize costs and token usage with charts
- Inspect the raw JSON responses from each model

## Directory Structure

```
token-poc/
├── data/
│   └── rubber_industry_transcript.txt  # Example transcript
├── prompts/
│   └── loan_eval_prompt.txt            # Prompt template
├── runs/
│   └── [timestamp]_summary.xlsx        # Generated results
├── driver.py                           # CLI batch runner
├── app.py                              # Streamlit dashboard
├── config.yaml.example                 # Example configuration
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## Customization

### Adding New Transcripts

Place additional transcript files in the `data/` directory and modify `driver.py` to process them.

### Modifying the Prompt

Edit `prompts/loan_eval_prompt.txt` to change the evaluation prompt.

### Adding New Models

Update `config.yaml` to include additional models from supported providers.

## Troubleshooting

- **API Key Issues**: Ensure your API keys are correctly set in `config.yaml`.
- **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`.
- **JSON Parsing Errors**: If models return non-JSON responses, they will be flagged in the results.

## License

[Specify license information here] 