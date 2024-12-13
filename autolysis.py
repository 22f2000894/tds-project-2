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
# ]
# ///

import os
import sys
import json
import hashlib
import pandas as pd
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

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "")



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
                self.session.cache.delete(expired=expired)
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

def get_hash_key(key_name):
    return hashlib.md5(key_name.encode('utf-8')).hexdigest()

# detect encoding of csv file
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    f.close()
    return result['encoding']



def get_column_details(filename, **kwargs):
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
    functions = {
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

    def number_of_lines(file_name):
        with open(file_name, 'r',encoding=encoding) as f:
            number_of_lines = len(f.readlines())
            f.close()
        return number_of_lines

    data = ""
    with open(filename, 'r',encoding=encoding) as f:
        for i in range(min(10, number_of_lines(filename))):
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
        "functions": [functions],
        "function_call": {
            "name": "get_column_type"
        }
    }

    try:

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



def column_name_for_correlation_matrix(dict_data):
    """
    Extracts a list of column names from a dictionary containing column metadata.

    Args:
        dict_data (dict): A dictionary containing column metadata.

    Returns:
        list: A list of column names.
    """
    col_list = []
    for col_dict in dict_data['column_metadata']:
        if col_dict['value_categories']['type'] == 'quantitative':
            col_list.append(col_dict['column_name'])
    return col_list

def create_correlation_matrix(df,selected_columns):
    """
    A function to create a correlation matrix from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to create the correlation matrix from.
        selected_columns (list): A list of column names to include in the correlation matrix.

    Returns:
        pandas.DataFrame: The correlation matrix.
    """
    correlation_matrix = df[selected_columns].corr()
    return correlation_matrix

# create heatmap using correlation matrix using seaborn
def create_heatmap(correlation_matrix, output_filename, **kwargs):
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


def embedding_image(image_path):
    # Path to your PNG file

    # Read and encode the image in Base64
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Create the Markdown content
    markdown_content = f"![Embedded Image](data:image/png;base64,{base64_string})"

    return markdown_content




# writing summery statistics take filename from command arguments
# Writing summary statistics - takes filename from command arguments
def summary_statistics(filename):
    try:
        # Detect encoding
        encoding = detect_encoding(filename)
        print("Detected encoding:", encoding)

        # Read CSV file
        data = pd.read_csv(filename, encoding=encoding)
        
        # clean null values
        data = data.dropna()

        # Get column details
        column_metadata = get_column_details(filename, encoding=encoding)
        
        # Create correlation matrix
        selected_columns = column_name_for_correlation_matrix(column_metadata)
        correlation_matrix = create_correlation_matrix(data, selected_columns)

        # Save summary statistics as Markdown
        file_directory = os.path.splitext(filename)[0]
        os.makedirs(file_directory, exist_ok=True)
        output_file = os.path.join(file_directory, f'{file_directory}.md')

        # correlation matrix heatmap
        title = f"{file_directory.upper()} Correlation Matrix Heatmap"
        heat_map_file = os.path.join(file_directory, f'{file_directory}_heatmap.png')
        create_heatmap(correlation_matrix, output_filename=heat_map_file, title=title)
      
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("## Summary Statistics\n")
            f.write(data.describe(include="all").to_markdown() + "\n")
            f.write("\n## Correlation Matrix\n")
            f.write(correlation_matrix.to_markdown() + "\n\n")
            f.write(embedding_image(heat_map_file))
            f.write("\n")
        
        f.close()
        print("Summary statistics saved to:", output_file)

    except Exception as e:
        print("An error occurred:", e)



# Command-line argument handling
if len(sys.argv) > 1:
    filename = sys.argv[1]
    summary_statistics(filename)
else:
    print("Please provide a filename")