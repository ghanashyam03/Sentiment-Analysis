import sys
import json

# Get the comment from the command-line argument
comment = sys.argv[1]

# Prepare the result as a dictionary
result = {
    'sentiment': comment
}

# Convert the result to JSON and print it
print(json.dumps(result))
