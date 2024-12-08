# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "openai",
# ]
# ///

import os
import sys
import pandas as pd
from openai import OpenAI
def print_hello_world():
    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN", "Hello World")
    print(AIPROXY_TOKEN)

print("Hello World")
print_hello_world()

# writing summery statistics take filename from command arguments
def summery_statistics(filename):
    data = pd.read_csv(filename, encoding='utf-8')
    null_values = data.isnull().sum()
    numeric_data = data.select_dtypes(include=['number'])

    correlation_matrix = numeric_data.corr()
    print("Null values:", null_values)
    print(data.describe(include="all"))
    print(correlation_matrix)

if len(sys.argv) > 1:
    filename = sys.argv[1]
    summery_statistics(filename)
    # # write markdown file of summery statistics
    # with open("summery_statistics.md", "w") as f:
    #     f.write(print(summery_statistics(filename)))
else:
    print("Please provide a filename")