#!/usr/bin/env python3
"""
Token-Economics PoC - Driver Script
Processes a transcript with various LLMs and records token usage and costs.
"""

import os
import sys
import json
import time
import yaml
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

# LangChain imports - using the newest structure (v0.3)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    """Configuration for a language model"""
    provider: str
    model_id: str
    model_name: str
    prompt_price: float  # per 1k tokens
    completion_price: float  # per 1k tokens

class Config(BaseModel):
    """Application configuration"""
    api_keys: Dict[str, str]
    models: Dict[str, Dict[str, str]]
    prices: Dict[str, Dict[str, Dict[str, float]]]
    max_retries: int = 3
    backoff_factor: int = 2
    timeout: int = 60

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)

def get_model_configs(config: Config) -> List[ModelConfig]:
    """Generate model configurations from the config file"""
    model_configs = []
    
    for provider, models in config.models.items():
        for model_id, model_name in models.items():
            model_configs.append(
                ModelConfig(
                    provider=provider,
                    model_id=model_id,
                    model_name=model_name,
                    prompt_price=config.prices[provider][model_id]["prompt"],
                    completion_price=config.prices[provider][model_id]["completion"]
                )
            )
    
    return model_configs

def get_chat_model(config: Config, model_config: ModelConfig):
    """Create and configure a chat model based on provider and model ID"""
    if model_config.provider == "openai":
        # o3 model doesn't support temperature=0.0, only default (1.0)
        temperature = 1.0 if model_config.model_id == "o3" else 0.0
        return ChatOpenAI(
            openai_api_key=config.api_keys["openai"],
            model=model_config.model_name,
            temperature=temperature,
            timeout=config.timeout
        )
    elif model_config.provider == "groqcloud":
        # Make sure we're using supported Groq models
        # Available models: llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768, gemma-7b-it
        logger.info(f"Using Groq model: {model_config.model_name}")
        
        return ChatGroq(
            groq_api_key=config.api_keys["groq"],
            model_name=model_config.model_name,
            temperature=0.0,
            timeout=config.timeout
        )
    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")

async def process_model(
    model_config: ModelConfig, 
    config: Config, 
    transcript: str, 
    prompt_template: str
) -> Dict[str, Any]:
    """Process a single model and return token usage and response data"""
    model_info = f"{model_config.provider}/{model_config.model_id}"
    logger.info(f"Processing model: {model_info}")
    
    # Make sure the transcript is properly incorporated into the prompt
    # Replace placeholder approach with more explicit format
    if "{{transcript}}" in prompt_template:
        formatted_prompt = prompt_template.replace("{{transcript}}", transcript)
    else:
        # Insert transcript explicitly before output format instructions
        # Find the output format section
        output_section_marker = "Your Output:"
        if output_section_marker in prompt_template:
            # Split the prompt at the output section
            parts = prompt_template.split(output_section_marker, 1)
            # Insert the transcript between instruction and output format
            formatted_prompt = f"{parts[0]}\n\nTRANSCRIPT:\n{transcript}\n\n{output_section_marker}{parts[1]}"
        else:
            # Fallback - just append transcript to the end of instructions
            formatted_prompt = f"{prompt_template}\n\nTRANSCRIPT:\n{transcript}"
    
    # Add explicit JSON formatting instructions to help guide the model
    if "JSON" in formatted_prompt and "{" in formatted_prompt and "}" in formatted_prompt:
        # The prompt already contains JSON instructions, no need to modify
        pass
    else:
        # Add explicit JSON formatting instructions
        formatted_prompt += "\n\nIMPORTANT: Your response must be valid JSON only, with no additional text before or after."
    
    # Log a preview of the formatted prompt
    preview_length = min(200, len(formatted_prompt))
    logger.info(f"Prompt preview for {model_info}: {formatted_prompt[:preview_length]}...")
    
    try:
        chat_model = get_chat_model(config, model_config)
        
        # Create a message to send to the model
        message = [HumanMessage(content=formatted_prompt)]
        
        # Call the model with retry logic
        response = None
        attempts = 0
        elapsed_time = 0
        
        while attempts < config.max_retries:
            try:
                start_time = time.time()
                response = await chat_model.ainvoke(message)
                elapsed_time = time.time() - start_time
                break
            except Exception as e:
                attempts += 1
                logger.warning(f"Attempt {attempts} failed for {model_info}: {str(e)}")
                if attempts < config.max_retries:
                    wait_time = config.backoff_factor ** attempts
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {model_info}")
                    raise
        
        # Get the model's actual response text
        response_text = response.content
        
        # Log a preview of the response for debugging
        response_preview = response_text[:200] if len(response_text) > 200 else response_text
        logger.info(f"Response preview from {model_info}: {response_preview}...")
        
        # Enhanced JSON extraction
        json_valid = True
        json_response = response_text
        
        try:
            # Step 1: Try direct parsing
            try:
                json_response = json.loads(response_text)
                logger.info(f"Successfully parsed JSON response from {model_info}")
            except json.JSONDecodeError:
                # Step 2: Try to extract JSON from text (between curly braces)
                import re
                
                # Look for any JSON-like pattern in the response
                json_pattern = r'(?s)\{.*\}'
                match = re.search(json_pattern, response_text)
                
                if match:
                    potential_json = match.group(0)
                    try:
                        json_response = json.loads(potential_json)
                        logger.info(f"Extracted JSON using simple pattern match for {model_info}")
                    except json.JSONDecodeError:
                        # Step 3: Try more complex pattern matching
                        # This handles nested structures better
                        try:
                            # Remove any markdown code block markers
                            cleaned_text = re.sub(r'```(?:json)?|```', '', response_text)
                            
                            # Find the outermost JSON object
                            json_pattern = r'(\{(?:[^{}]|(?1))*\})'
                            matches = re.findall(json_pattern, cleaned_text)
                            
                            if matches:
                                for potential_match in matches:
                                    try:
                                        json_response = json.loads(potential_match)
                                        logger.info(f"Extracted JSON using recursive regex for {model_info}")
                                        break
                                    except:
                                        continue
                            else:
                                raise Exception("No valid JSON pattern found")
                        except:
                            # If all parsing fails, we'll use the text as is
                            logger.warning(f"Could not extract valid JSON for {model_info}")
                            json_valid = False
                else:
                    logger.warning(f"No JSON-like pattern found in {model_info} response")
                    json_valid = False
            
            # Verify the expected structure exists
            expected_keys = ["parameter_evaluations", "overall_decision", "key_reasons"]
            
            if json_valid and isinstance(json_response, dict):
                missing_keys = [key for key in expected_keys if key not in json_response]
                if missing_keys:
                    logger.warning(f"JSON response from {model_info} is missing expected keys: {missing_keys}")
                    # We'll still use the JSON, just with a warning
            
        except Exception as e:
            logger.warning(f"Error parsing JSON response from {model_info}: {str(e)}")
            json_valid = False
        
        # Token counting
        if hasattr(response, 'usage') and response.usage:
            # Some models return usage info directly
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        else:
            # Otherwise use LangChain's token counting
            input_tokens = chat_model.get_num_tokens(formatted_prompt)
            output_tokens = chat_model.get_num_tokens(response_text)
        
        total_tokens = input_tokens + output_tokens
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * model_config.prompt_price
        output_cost = (output_tokens / 1000) * model_config.completion_price
        total_cost = input_cost + output_cost
        
        return {
            "model": model_info,
            "status": "success",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "elapsed_seconds": elapsed_time,
            "json_valid": json_valid,
            "json_response": json_response if isinstance(json_response, dict) else response_text
        }
    
    except Exception as e:
        logger.error(f"Error processing {model_info}: {str(e)}")
        return {
            "model": model_info,
            "status": "error",
            "error_message": str(e),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "elapsed_seconds": 0.0,
            "json_valid": False,
            "json_response": str(e)
        }

async def run_all_models(config: Config, transcript_path: str, prompt_path: str) -> pd.DataFrame:
    """Run all models and collect results in a DataFrame"""
    # Load transcript and prompt
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    
    # Log the transcript being used
    logger.info(f"Using transcript from: {transcript_path}")
    
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    model_configs = get_model_configs(config)
    
    # Process all models concurrently
    tasks = [
        process_model(model_config, config, transcript, prompt_template)
        for model_config in model_configs
    ]
    
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

def save_results(df: pd.DataFrame, run_id: str, transcript_path: str) -> str:
    """Save results to an Excel file"""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Extract transcript filename without extension
    transcript_name = Path(transcript_path).stem
    
    output_path = runs_dir / f"{transcript_name}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_summary.xlsx"
    
    # Prepare data for Excel
    # Convert complex JSON to strings
    df_excel = df.copy()
    df_excel["json_response"] = df_excel["json_response"].apply(
        lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x)
    )
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_excel.to_excel(writer, sheet_name=run_id, index=False)
        
        # Auto-adjust column widths
        for column in df_excel:
            column_width = max(df_excel[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = df_excel.columns.get_loc(column)
            writer.sheets[run_id].column_dimensions[chr(65 + col_idx)].width = min(column_width, 50)
    
    return str(output_path)

async def main():
    """Main entry point for the script"""
    # Generate timestamp for this run
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger.info(f"Starting run: {run_id}")
    
    # Check for config file
    if not os.path.exists("config.yaml"):
        logger.error("Config file not found. Please copy config.yaml.example to config.yaml and update it.")
        sys.exit(1)
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Path to prompt
    prompt_path = "prompts/loan_eval_prompt.txt"
    logger.info(f"Using prompt: {prompt_path}")
    
    # Get all transcript files from data directory
    data_dir = Path("data")
    transcript_files = list(data_dir.glob("*.txt"))
    
    if not transcript_files:
        logger.error("No transcript files found in the data directory.")
        sys.exit(1)
    
    logger.info(f"Found {len(transcript_files)} transcript files to process.")
    
    last_output_path = None
    
    # Process each transcript file
    for i, transcript_path in enumerate(transcript_files):
        # Add delay between files (except before the first one)
        if i > 0:
            logger.info(f"Waiting 1 minute before processing next file...")
            await asyncio.sleep(60)
            
        logger.info(f"\nProcessing transcript: {transcript_path}")
        
        # Run all models for this transcript
        results_df = await run_all_models(config, str(transcript_path), prompt_path)
        
        # Save individual results
        output_path = save_results(results_df, run_id, str(transcript_path))
        logger.info(f"Results for {transcript_path.name} saved to: {output_path}")
        
        # Store the last output path
        last_output_path = output_path
    
    return last_output_path

if __name__ == "__main__":
    # Run the async main function
    output_path = asyncio.run(main()) 