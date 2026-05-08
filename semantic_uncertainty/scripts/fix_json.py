import glob
import re
import ast
import json

files = glob.glob("/home/kalashkala/Datasets/Semantic-Uncertainty/sample-data/*__mcq/*.json")
for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    def repl(match):
        list_str = f"[{match.group(1)}]"
        try:
            val = ast.literal_eval(list_str)
            return f'"ground_truth": {json.dumps(val)}'
        except Exception as e:
            print(f"Error on {list_str}: {e}")
            return match.group(0)
            
    s = re.sub(r'"ground_truth":\s*\[(.*?)\]', repl, content)
    
    with open(f, "w") as file:
        file.write(s)
        
    # Test if it loads now
    try:
        json.loads(s)
        print(f"Successfully fixed and loaded {f}")
    except Exception as e:
        print(f"Still failing to load {f}: {e}")
