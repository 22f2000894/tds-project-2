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
# ]
# ///

import os
import sys
import pandas as pd
import tabulate
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
def print_hello_world():
    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "Hello World")
    print(AIPROXY_TOKEN)

print("Hello World")
print_hello_world()

# detect encoding of csv file
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


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
    pass


# Command-line argument handling
if len(sys.argv) > 1:
    filename = sys.argv[1]
    summary_statistics(filename)
else:
    print("Please provide a filename")