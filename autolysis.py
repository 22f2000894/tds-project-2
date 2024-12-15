# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "openai",
#     "tabulate",
#     "chardet",
#     "seaborn",
#     "tenacity",
#     "requests_cache",
#     "requests",
#     "scipy",
# ]
# ///

import sys
import os
import json
import hashlib
import pandas as pd
import numpy as np
import tabulate
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
import requests_cache
import requests
import pytz
import base64
from openai import OpenAI
from datetime import timedelta
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from scipy import stats

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "")

url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization" : f"Bearer {AIPROXY_TOKEN}"
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))  # Retry 3 times, wait 2 seconds between attempts
def make_api_call(url, **kwargs):
    """
    Make an HTTP request
    :kwargs: headers, json_data, params, url, data, timeout

    """
    method = kwargs.get('method', 'POST')
    json_data = kwargs.get('json_data', {})
    data = kwargs.get('data', None)
    params = kwargs.get('params', None)
    headers = kwargs.get('headers', None)
    timeout = kwargs.get('timeout', 10)
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, timeot=timeout)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, json=json_data, timeout=timeout)
        else:
            raise ValueError("Unsupported HTTP method")

        # Raise an exception for HTTP errors
        response.raise_for_status()
        return response

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise  # Propagate the exception to trigger retry



def call_api_with_retry(url, **kwargs):

    method = kwargs.get('method', 'POST')
    headers = kwargs.get('headers', None)
    json_data = kwargs.get('json_data', {})
    timeout = kwargs.get('timeout', 10)

    try:
        response = make_api_call(url, method=method, json_data=json_data, headers=headers, timeout=timeout)
        return response
    except RetryError as e:  # Catches when all retries fail
        print(f"All retry attempts failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


class CustomCacheManager:
    def __init__(self, cache_name='api_cache',**kwargs):
        """
        Initialize a cached session with custom cache key management
        """
        self.expire_after = kwargs.get('expire_after', timedelta(hours=24))
        tz = pytz.timezone("Asia/Kolkata")
        self.absolute_expiry = datetime.now(tz=tz) + self.expire_after
        # Create a cached session
        self.session = requests_cache.CachedSession(
            cache_name=cache_name,
            backend='sqlite',  # Using SQLite as cache backend
            expire_after= self.expire_after,  # Cache expiration
            allowable_methods=('GET', 'HEAD','OPTIONS','POST','PUT','DELETE', 'PATCH'),
            cache_control=True
        )

    def set_cache_response(self, cache_key, url, **kwargs):
        """
        Manually set a cached response with a specific cache key
        
        :param cache_key: Custom unique identifier for the cache entry
        :param url: URL to fetch
        :param params: Optional query parameters
        :param method: HTTP method 
        :return: Response object
        """
        # Perform the 
        method = kwargs.get('method', 'POST')
        params = kwargs.get('params', None)
        headers = kwargs.get('headers', None)
        json_data = kwargs.get('json_data', {})
        url = url
        
        response = call_api_with_retry(url, method=method, headers=headers, json_data=json_data)
        if response == False:
            return None
        else:
            # Manually set the cache key
            self.session.cache.save_response(
                cache_key=cache_key,  # Use the custom cache key
                response=response,
                expires = self.absolute_expiry
            )

            self.session.cache.close()

            return response


       

    def get_cached_response(self, cache_key):
        """
        Retrieve a cached response by its custom cache key
        
        :param cache_key: Custom cache key to retrieve
        :return: Cached response or None
        """
        try:
            # Attempt to retrieve the cached response
            cached_response = self.session.cache.get_response(key=cache_key)
            if cached_response:
                return cached_response
        except KeyError:
            print(f"No cached response found for key: {cache_key}")
            return None

    def delete_cache_entry(self,**kwargs):
        """
        Delete a specific cache entry by its cache key
        
        :param cache_key: Custom cache key to delete
        :return: Boolean indicating success
        """
        try:
            cache_key = kwargs.get('cache_key',None)
            expired = kwargs.get('expired', True)
            # Remove the specific cache 
            if cache_key is None:
                print("No cache key provided. Please provide a valid cache key.")
                return False
            else:
                self.session.cache.delete(cache_key)
                print(f"Cache entry deleted for key: {cache_key}")

            return True
        except Exception as e:
            print(f"Error deleting cache entry: {e}")
            return False

    def clear_cache(self):
        """
        Clear the entire cache
        """
        self.session.cache.clear()

        print("Cache cleared.")

    def remove_expired_cache(self):
        """
        Remove expired cache entries
        """
        self.session.cache.delete(expired = True)

        print("Expired cache entries removed.")
    
    

cache_manager = CustomCacheManager(cache_name='api_cache', expire_after=timedelta(hours=24))

cache_manager.remove_expired_cache()

# ---------------------------------------------
# class DataAnalysisAssistant to analyze data and get suggestions of columns
# ---------------------------------------------

class DataAnalysisAssistant:
    def __init__(self, cache_manager=None, url=None, headers=None):
        """
        Initialize the Data Analysis Assistant with optional caching and API configurations.

        Args:
            cache_manager: A cache management system for storing API responses
            url: API endpoint URL
            headers: API request headers
        """
        self.cache_manager = cache_manager
        self.url = url
        self.headers = headers or {}
        
        # Define the system prompt as a class attribute for better organization
        self.system_prompt = """
        You are a data analysis assistant designed to parse column name and its data type 
        for performing data analysis based on the structure and metadata of a CSV dataset. 
        The dataset's metadata includes column names, data types, and value categories such 
        as quantitative/qualitative and their subcategories (e.g., nominal, ordinal, discrete, continuous). 
        
        Key Analysis Focus Areas:
        1. Summary Statistics
        2. Missing Value Analysis
        3. Correlation Matrices
        4. Outlier Detection
        5. Clustering
        6. Hierarchy Detection
        7. Advanced Analyses:
           - Group-wise Summaries
           - Feature Importance
           - Data Visualization
        8. Potential Insights:
           - Outlier Detection: Identify errors, fraud, or opportunities
           - Correlation Analysis: Understand impact factors
           - Time Series Analysis: Predict future patterns
           - Cluster Analysis: Identify natural groupings
           - Network Analysis: Discover cross-selling or collaboration opportunities
        """

    def _generate_hash_key(self, key_name: str) -> str:
        """
        Generate a hash key for caching.

        Args:
            key_name: A string to be hashed

        Returns:
            A hash key as a string
        """
        return hashlib.md5(key_name.encode()).hexdigest()

    def _prepare_function_descriptions(self) -> List[Dict[str, Any]]:
        """
        Prepare function descriptions for API call.

        Returns:
            A list of function description dictionaries
        """
        return [
            {
                "name": "get_data_analysis_functions_with_parameters",
                "description": "Choose appropriate columns from metadata based on analysis function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "outlier_and_anomaly_detection": {
                            "type": "array",
                            "description": "Columns suitable for outlier detection",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column_name": {
                                        "type": "string",
                                        "description": "Name of column for outlier analysis"
                                    },
                                    "data_type": {
                                        "type": "string",
                                        "description": "Data type of column"
                                    }
                                }
                            }
                        },
                        "create_correlation_matrix": {
                            "type": "array",
                            "description": "Columns suitable for correlation matrix",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column_name": {
                                        "type": "string",
                                        "description": "Name of column for correlation analysis"
                                    },
                                    "data_type": {
                                        "type": "string",
                                        "description": "Data type of column"
                                    }
                                }
                            }
                        },
                        "time_series_analysis": {
                            "type": "array",
                            "description": "Columns suitable for time series analysis",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column_name": {
                                        "type": "string",
                                        "description": "Name of column for time series analysis"
                                    },
                                    "data_type": {
                                        "type": "string",
                                        "description": "Data type of column. Can be 'date', 'datetime', or 'time' and quantitative column also has to be present."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ]

    def data_analysis(
        self, 
        column_metadata: str, 
        model: str = "gpt-4o-mini", 
        filename: str = ""
    ) -> Dict[str, Any]:
        """
        Perform data analysis based on column metadata.

        Args:
            column_metadata: Metadata about dataset columns
            model: AI model to use for analysis
            filename: Name of the file being analyzed

        Returns:
            Dictionary of analysis results
        """
        # Prepare messages for API call
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"""Suggest list of column names for analysis functions:
                - Outlier and Anomaly Detection
                - Correlation Matrix
                - Time Series Analysis
                Column Metadata: {column_metadata}"""
            }
        ]

        # Prepare JSON payload for API request
        json_data = {
            "model": model,
            "messages": messages,
            "functions": self._prepare_function_descriptions(),
            "function_call": {
                "name": "get_data_analysis_functions_with_parameters"
            }
        }

        try:
            # Generate cache key
            cache_key = self._generate_hash_key(f"data_analysis_{filename}")
            
            # Check and manage cache
            if self.cache_manager:
                response = self.cache_manager.get_cached_response(cache_key)
                if response is None:
                    response = self.cache_manager.set_cache_response(
                        cache_key = cache_key,
                        url = self.url,
                        json_data = json_data,
                        headers = self.headers 
                    )
                    if response is None:
                        raise Exception("API request failed")
                    function_call = response.json()['choices'][0]['message']['function_call']
                    return json.loads(function_call['arguments'])
                            
            
                # Process and return response
                elif response:
                    function_call = response.json()['choices'][0]['message']['function_call']
                    return json.loads(function_call['arguments'])
                
        except Exception as e:
            print(f"Data Analysis Error: {e}")
            return {}

        
    def encode_image(self, image_path):
        # Path to your PNG file

        # Read and encode the image in Base64
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_string

    def narate_story_of_chart(self, image_path, **kwargs):
        """
        Generate narrative of the chart image
        """
        timeout = kwargs.get('timeout', 20)
        model = kwargs.get('model', 'gpt-4o-mini')
        system_prompt = """
        You are a data analysis assistant designed to generate a narrative summary of a chart image.
        """
        user_prompt = """
        Please generate a brief summary of the following chart image.
        """
        # encoding image
        image_string = self.encode_image(image_path)
        image_name = os.path.basename(image_path)
        
        # Prepare messages for API call
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": user_prompt
                    },
                    {
                        "type": "image_url", 
                        "image_url": {
                                        "url": f"data:image/png;base64,{image_string}",
                                        "detail": "low"             
                                    }
                    }
                ]
            }
        ]

        # Prepare JSON payload for API request
        json_data = {
            "model": model,
            "messages": messages,
            "max_tokens": 300
        }

        try:
            # Generate cache key
            cache_key = self._generate_hash_key(f"narate_story_of_chart_{image_name}")
            
            # Check and manage cache
            if self.cache_manager:
                response = self.cache_manager.get_cached_response(cache_key)
                if response is None:
                    response = self.cache_manager.set_cache_response(
                        cache_key = cache_key,
                        url = self.url,
                        json_data = json_data,
                        headers = self.headers,
                        timeout = timeout 
                    )
                    if response is None:
                        raise Exception("API request failed")
                    return response.json()['choices'][0]['message']['content']
                            
            
                # Process and return response
                elif response:
                    return response.json()['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"Data Analysis Error: {e}")
            return ""

    def narate_summary(self,filename, **kwargs):
        """
        Generate summary of the dataset
        """
        summary_details = kwargs.get('summary_details', "")
        system_prompt = """
            You are a data analysis assistant designed to generate a summary of a dataset.
        """
        user_prompt = f"""
            Please generate a brief summary of the following {summary_details}.
        """
        # Prepare messages for API call
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # Prepare JSON payload for API request
        json_data = {
            "model": "gpt-4o-mini",
            "messages": messages,
        }
    
        try:

            # Generate cache key
            cache_key = self._generate_hash_key(f"narate_summary_{filename}")
            
            # Check and manage cache
            if self.cache_manager:
                response = self.cache_manager.get_cached_response(cache_key)
                if response is None:
                    response = self.cache_manager.set_cache_response(
                        cache_key = cache_key,
                        url = self.url,
                        json_data = json_data,
                        headers = self.headers 
                    )
                    if response is None:
                        raise Exception("API request failed")
                    return response.json()['choices'][0]['message']['content']
                            
            
                # Process and return response
                elif response:
                    return response.json()['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"Data Analysis Error: {e}")
            return ""

# ---------------------------------------------
# class AdvanceDataAnalysis  to analyze data
# ---------------------------------------------

class AdvancedDataAnalysis():
    def __init__(self, csv_file_path):
        """
        Initialize the analysis with a CSV file
        
        Args:
            csv_file_path (str): Path to the CSV file
        """
        # Load the data
        encoding = detect_encoding(csv_file_path)
        self.df = pd.read_csv(csv_file_path, encoding=encoding)
        
        # Prepare output directory
        filename = os.path.basename(csv_file_path)
        file_directory = os.path.splitext(filename)[0]
        os.makedirs(file_directory, exist_ok=True)

    def outlier_and_anomaly_detection(self, method='iqr', **kwargs):
    
        """
        Detect outliers using multiple methods
        
        Args:
            method (str): Method for outlier detection ('iqr', 'zscore' )
        
        Returns:
            dict: Outlier detection results
        """
        selected_columns = kwargs.get('selected_columns', [])
        # Numerical columns only
        outliers = {}

        if selected_columns:
            numerical_cols = selected_columns
        else:
            numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Detect outliers
                column_outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                
                outliers[col] = {
                    'total_outliers': len(column_outliers),
                    'outlier_percentage': len(column_outliers) / len(self.df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        elif method == 'zscore':
            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(self.df[col]))
                column_outliers = self.df[z_scores > 3]
                
                outliers[col] = {
                    'total_outliers': len(column_outliers),
                    'outlier_percentage': len(column_outliers) / len(self.df) * 100
                }
        
        
        return outliers

    def create_correlation_matrix(self, selected_columns):
        """
        A function to create a correlation matrix from a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to create the correlation matrix from.
            selected_columns (list): A list of column names to include in the correlation matrix.

        Returns:
            pandas.DataFrame: The correlation matrix.
        """
        if selected_columns:
            correlation_matrix = self.df[selected_columns].corr()
            return correlation_matrix
        else:
            numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
            correlation_matrix = self.df[numerical_cols].corr()
            return correlation_matrix

    # create heatmap using correlation matrix using seaborn
    def create_heatmap(self, correlation_matrix, output_filename, **kwargs):
        """
        Create and save a heatmap from a correlation matrix.

        Parameters:
            correlation_matrix (pd.DataFrame): The correlation matrix.
            output_file (str): The path to save the heatmap PNG file.

        Returns:
            None
        """
        # Set the output file path
        output_filename = output_filename
        title = kwargs.get('title', 'Correlation Matrix Heatmap')

        # Set the figure size
        plt.figure(figsize=(10, 8))
        
        # Create the heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,         # Show correlation values on the heatmap
            cmap="coolwarm",    # Choose a colormap
            fmt=".2f",          # Format for annotation
            linewidths=0.5,     # Line width between cells
        )
        
        # Add title to the heatmap
        plt.title(title, fontsize=16)
        
        # Save the heatmap as a PNG file
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        
        # Close the plot to avoid display issues in interactive environments
        plt.close()


 
    def time_series_analysis(self, file_path, selected_columns, output_file_path, **kwargs):
        # Get method, model, and period from kwargs
        method = kwargs.get('method', 'ffill')
        model = kwargs.get('model', 'additive')
        period = kwargs.get('period', 12)  # Default period is 12 (monthly data)
        datetime_column = kwargs.get('datetime_column', None)

        # Read data from CSV
        df = pd.read_csv(file_path, parse_dates=[datetime_column], index_col=datetime_column)
        
        # Clean the data for each selected column
        for column in selected_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, handling errors
            df[column] = df[column].fillna(method=method)  # Fill missing values
        
        # Loop over selected columns for seasonal decomposition
        for column in selected_columns:
            try:
                # Perform seasonal decomposition on the current column
                result = seasonal_decompose(df[column], model=model, period=period)

                # Decide the best model (additive or multiplicative)
                decision = decide_model(result)
                
                # Re-decompose if needed (multiplicative model)
                if decision == "multiplicative":
                    result = seasonal_decompose(df[column], model="multiplicative", period=period)
                
                # Plot the decomposition results
                result.plot()
                plt.savefig(f"{output_file_path}_{column}.png")
                plt.close()  # Close the plot after saving it

            except ValueError as e:
                print(f"Error with column {column}: {e}")
                continue

            def decide_model(result):
                seasonal = result.seasonal
                trend = result.trend
                residual = result.resid

                # Check 1: Seasonal range proportionality
                seasonal_range = seasonal.max() - seasonal.min()
                mean_trend = trend.mean()
                proportionality = seasonal_range / mean_trend

                # Check 2: Residual coefficient of variation
                residual_cv = np.std(residual.dropna()) / np.mean(trend.dropna())

                # Check 3: Seasonal-to-trend ratio variance
                seasonal_to_trend_ratio = (seasonal / trend).dropna()
                ratio_variance = np.var(seasonal_to_trend_ratio)

                # Decision rules
                if proportionality > 0.1 or residual_cv > 0.1 or ratio_variance > 0.05:
                    return "multiplicative"
                else:
                    return "additive"




def get_hash_key(key_name):
    return hashlib.md5(key_name.encode('utf-8')).hexdigest()

# detect encoding of csv file
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    f.close()
    return result['encoding']



def get_column_details(file_path, **kwargs):
    # declare AIPROXY_TOKEN as global variable
    global AIPROXY_TOKEN
    # getting file encoding 
    encoding = kwargs.get('encoding', 'utf-8')
    # openai model
    model = kwargs.get('model', 'gpt-4o-mini')
    # url to do post request
    url = kwargs.get('url', 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions')

    AIPROXY_TOKEN = kwargs.get('AIPROXY_TOKEN', AIPROXY_TOKEN)

    # headers
    headers = {
        "Content-Type": "application/json",
        "Authorization" : f"Bearer {AIPROXY_TOKEN}"
    }

    # system prompt 
    content = """
        Analyze the given dataset. The first line represents the header, and subsequent lines provide sample data. Columns may contain uncleaned or inconsistent data; ignore such cells when determining column properties. Infer the data type of each column by considering the majority of valid values, which can be one of the following: 'integer', 'float', 'string', 'date', 'datetime', or 'boolean'. Additionally, classify each column as 'quantitative' or 'qualitative', and further categorize them into subcategories: 'discrete', 'continuous', 'nominal', or 'ordinal'.

        Handle time-related columns (e.g., years, dates, datetimes) carefully:
        - Classify a time column as 'qualitative ordinal' if it represents an ordered sequence (e.g., event timelines, publication years, or historical rankings).
        - Classify it as 'quantitative continuous' if it represents measurable intervals or durations (e.g., age in years, elapsed time).

        Special cases for identifier-like columns (e.g., IDs, codes, roll numbers, pincode, phone numbers etc):
        - Treat these as 'qualitative nominal' and 'qualitative ordinal', regardless of whether they appear alphanumeric and numeric respectively.
    """
    system_prompt = kwargs.get('system_prompt', content)

    # function call
    functions = [
        {
            "name": "get_column_type",
            "description": "Identify column names, their data types, and whether they contain quantitative or qualitative data from a dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column_metadata": {
                        "type": "array",
                        "description": "Metadata for each column in the dataset.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column_name": {
                                    "type": "string",
                                    "description": "The name of the column."
                                },
                                "column_type": {
                                    "type": "string",
                                    "description": "The data type of the column (e.g., integer, float, string, etc.). If mixed data types are present in the column, determine the majority type."
                                },
                                "value_categories": {
                                    "type": "object",
                                    "description": "Specifies if the column contains quantitative or qualitative data, along with a subcategory.",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "description": "Whether the data is quantitative or qualitative."
                                        },
                                        "subcategory": {
                                            "type": "string",
                                            "description": "The subcategory of the data type (e.g., nominal, ordinal, discrete, continuous)."
                                        }
                                    },
                                    "required": ["type", "subcategory"]
                                }
                            },
                            "required": ["column_name", "column_type", "value_categories"]
                        },
                        "minItems": 1
                    }
                },
                "required": ["column_metadata"]
            }
        }
    ]

    def number_of_lines(file_path):
        with open(file_path, 'r', encoding=encoding) as f:
            number_of_lines = len(f.readlines())
            f.close()
        return number_of_lines

    data = ""
    with open(file_path, 'r', encoding=encoding) as f:
        for i in range(min(10, number_of_lines(file_path))):
            data = data + f.readline()
        f.close()

    user_prompt = kwargs.get('user_prompt', data)
    
    # json data pass to openai to get column name, column type and category of column
    json_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        "functions": functions,
        "function_call": {
            "name": "get_column_type"
        }
    }

    try:
        
        
        filename = os.path.basename(file_path)
        # hash key
        key_name = f"get_column_details_{filename}"
        cache_key = get_hash_key(key_name)
        # post request to get 
        
        response = cache_manager.get_cached_response(cache_key)
        if response is None:
            response = cache_manager.set_cache_response(cache_key, url, headers=headers, json_data=json_data)
        
        if response:
            column_metadata = json.loads(response.json()['choices'][0]['message']['function_call']['arguments'])
            return column_metadata
           
    except Exception as e:
        print(f"Error: {e}")
        return {}



def get_function_column_list(function_metadata, function_name):
    """
    Get column names for a specific function.

    Args:
        function_metadata (dict): Dictionary containing function metadata.
        function_name (str): Name of the function.

    Returns:
        List of column names for the specified function.
    """
    l = []
    for dict_data in function_metadata[function_name]:
        l.append(dict_data['column_name'])
    return l

def embedding_image(image_path):
    # Path to your PNG file

    # Read and encode the image in Base64
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the Markdown content
    markdown_content = f"![Embedded Image](data:image/png;base64,{base64_string})"

    return markdown_content

def make_markdown_file(file_path):
    """
    Create a Markdown file for a given filename and directory.

    Args:
        filename (str): Name of the file.
    
    """
    global url
    try:
        # Detect encoding
        encoding = detect_encoding(file_path)
        print("Detected encoding:", encoding)

        # Read CSV file
        data = pd.read_csv(file_path, encoding=encoding)
        
        # Save summary statistics as Markdown
        filename = os.path.basename(file_path)
        file_directory = os.path.splitext(filename)[0]
        os.makedirs(file_directory, exist_ok=True)
        output_file = os.path.join(file_directory, 'README.md')

        # Get column metadata

        # Get column details
        column_metadata = get_column_details(file_path, encoding=encoding)
        # print(column_metadata)
        # data 
        data_analyse = DataAnalysisAssistant(cache_manager=cache_manager, url=url, headers=headers)
        function_metadata = data_analyse.data_analysis(column_metadata, filename=filename)
        
        # advance analysis 
        advanced_analysis = AdvancedDataAnalysis(file_path)
        # outlier and anamaly detection based on zscore
        
        outliers_column_list = get_function_column_list(function_metadata, function_name='outlier_and_anomaly_detection')
        outliers = advanced_analysis.outlier_and_anomaly_detection(method='zscore', selected_columns=outliers_column_list)
        outliers_df = pd.DataFrame(outliers)
        # Create correlation matrix
        correlation_columns = get_function_column_list(function_metadata, function_name='create_correlation_matrix')
        correlation_matrix = advanced_analysis.create_correlation_matrix(selected_columns=correlation_columns)
        headmap_file = os.path.join(file_directory, f'{file_directory}_heatmap.png') 
        
        # Create heatmap
        title = f"{file_directory.upper()} Correlation Matrix Heatmap"
        advanced_analysis.create_heatmap(correlation_matrix, output_filename=headmap_file, title=title)
        headmap_image = embedding_image(headmap_file)

        # heatmap summary
        chart_summary = data_analyse.narate_story_of_chart(headmap_file)

        # data summary
        summary_details = "" +str(data.describe(include="all")) + str(outliers)
        dataset_summary = data_analyse.narate_summary(filename, summary_details=summary_details)

        # Save summary statistics as Markdown
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Data Analysis of csv file - " + filename + "\n\n")
            f.write("## Dataset\n\n")
            f.write(data.head().to_markdown() + "\n\n")
            f.write("## Summary Statistics\n\n")
            f.write(data.describe(include="all").to_markdown() + "\n\n")
            f.write("## Outlier And Anomaly Detection\n\n")
            f.write(outliers_df.to_markdown() + "\n\n")
            f.write(dataset_summary + "\n\n")
            f.write("## Correlation Matrix\n\n")
            f.write(correlation_matrix.to_markdown() + "\n\n")
            f.write(headmap_image + "\n\n")
            f.write(chart_summary + "\n\n")
            f.close()


    except Exception as e:
        print(f"Error: {e}")

# Command-line argument handling
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    print(file_path)
    filename = os.path.basename(file_path)
    print(filename)
    make_markdown_file(file_path)
else:
    print("Please provide a filename")