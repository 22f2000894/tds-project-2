# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "openai",
#     "tabulate",
#     "chardet",
#     "seaborn",
#     "tenacity"
#     "requests_cache"
#     "requests"
#     "hashlib"
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
from openai import OpenAI
def print_hello_world():
    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "Hello World")
    print(AIPROXY_TOKEN)

print("Hello World")
print_hello_world()


class CustomCacheManager:
    def __init__(self, cache_name='custom_cache'):
        """
        Initialize a cached session with custom cache key management
        """
        # Create a cached session
        self.session = requests_cache.CachedSession(
            cache_name=cache_name,
            backend='sqlite',  # Using SQLite as cache backend
            expire_after=timedelta(hours=24)  # Cache expiration
        )

    def set_cache_response(self, cache_key, url, params=None, method='GET'):
        """
        Manually set a cached response with a specific cache key
        
        :param cache_key: Custom unique identifier for the cache entry
        :param url: URL to fetch
        :param params: Optional query parameters
        :param method: HTTP method (default: GET)
        :return: Response object
        """
        # Perform the request
        # response = self.session.request(method, url, params=params)
        response = requests.get(url)
        # Manually set the cache key
        self.session.cache.save_response(
            cache_key=cache_key,  # Use the custom cache key
            response=response
        )
        
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
            return cached_response
        except KeyError:
            print(f"No cached response found for key: {cache_key}")
            return None

    def delete_cache_entry(self, cache_key):
        """
        Delete a specific cache entry by its cache key
        
        :param cache_key: Custom cache key to delete
        :return: Boolean indicating success
        """
        try:
            # Remove the specific cache entry
            self.session.cache.delete(cache_key)
            print(f"Cache entry deleted for key: {cache_key}")
            return True
        except Exception as e:
            print(f"Error deleting cache entry: {e}")
            return False






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
    encoding = detect_encoding(filename)
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
    with open(file_name, 'r',encoding=encoding) as f:
        for i in range(min(10, number_of_lines(file_name))):
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
        # post request to get column_metadata
        response = requests.post(url=url, headers=headers, json=json_data)
        column_metadata = json.loads(response.json()['choices'][0]['message']['function_call']['arguments'])

        return column_metadata
    except Exception as e:
        raise e



# create heatmap using correlation matrix using seaborn
def create_heatmap(correlation_matrix, output_file="correlation_heatmap.png"):
    """
    Create and save a heatmap from a correlation matrix.

    Parameters:
        correlation_matrix (pd.DataFrame): The correlation matrix.
        output_file (str): The path to save the heatmap PNG file.

    Returns:
        None
    """
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
    plt.title("Correlation Matrix Heatmap", fontsize=16)
    
    # Save the heatmap as a PNG file
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    # Close the plot to avoid display issues in interactive environments
    plt.close()

#  correlation matrix function
def create_correlation_matrix(df,selected_columns):
    correlation_matrix = df[selected_columns].corr()
    return correlation_matrix

# correlation matrix analysis 
def correlation_matrix_analysis(correlation_matrix):
    # generate story about correlation matrix using openai and heatmap of correlation matrix
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Generate a story about the correlation matrix: {correlation_matrix.to_markdown()}",
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )



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

        # Numeric data correlation matrix
        numeric_data = data.select_dtypes(include=['number'])
        correlation_matrix = numeric_data.corr()

        # Save summary statistics as Markdown
        file_directory = os.path.splitext(filename)[0]
        os.makedirs(file_directory, exist_ok=True)
        output_file = os.path.join(file_directory, f'{file_directory}.md')
        # correlation matrix heatmap
        heat_map_file = os.path.join(file_directory, f'{file_directory}_heatmap.png')
        create_heatmap(correlation_matrix, output_file=heat_map_file)
      
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("## Summary Statistics\n")
            f.write(data.describe(include="all").to_markdown() + "\n")
            f.write("\n## Correlation Matrix\n")
            f.write(correlation_matrix.to_markdown() + "\n")

        print("Summary statistics saved to:", output_file)

    except Exception as e:
        print("An error occurred:", e)


def function_calling(*args, **kwargs):

    json_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "give coulmn name and its type of dataframe"
            }
        ]
    }

    # content 
    content = {
        "Analyze the given dataset. The first line is header, and subsequent lines"
        "are sample data. Columns may have uncleaned data in them ignore those cells, Infer the data types by considering mazority of the data."
        "'integer', 'float', 'string', 'date', 'datetime', 'boolean'."
    }
    
    # function schema
    functions = {
        "name" : "get_column_type",
        "description": "Identify column names and their data types from a dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_metadata": {
                    "type": "array",
                    "description": "Meta data for each column.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column_name": {
                                "type": "string",
                                "description": "The Name of the column."
                            },
                            "column_type": {
                                "type": "string",
                                "description": "The data type of the column (e.g. integer, float, string, etc.)."
                            }
                        },
                        "required": ["column_name", "column_type"]
                    },
                    "minItems" : 1,
                },
            },
            "required": ["column_metadata"]
        }
    }
        


# Command-line argument handling
if len(sys.argv) > 1:
    filename = sys.argv[1]
    summary_statistics(filename)
else:
    print("Please provide a filename")