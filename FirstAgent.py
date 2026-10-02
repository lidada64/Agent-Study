import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
import logging
load_dotenv();

logger = logging.getLogger(__name__)
client=OpenAI()
#logging system config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",#format
    handlers=[
        logging.StreamHandler(),#print on termina; 
        logging.FileHandler("agent.log", encoding="utf-8"),#write in log file
    ],
)

#tool call "find file and find text" 
#find text: file
tools=[
    {
    "type":"function",
    "description":"Find one or more files by the name and return absolute path of file",
    "name":"find_file",

    "parameters":{
        "type":"object",
        "properties":{
            "name":{
            "type":"string",
            "description":"name of a file",
            },
        },
    "required":["name"],#Declare required at the same level as properties
    },

    },
     {
    "type":"function",
    "description":"If there are one or more file paths,Find the specific text location from the path motioned by customer",
    "name":"find_text",

    "parameters":{
        "type":"object",
        "properties":{
            "paths":{#paths is a list, not a single string
            "type":"array",
            "items":{"type":"string"},
            "description":"Absolute paths of file list",
            },
            "text":{
            "type":"string",
            "description":"Specific keywords defined by user"
            },
        },
        "required":["paths","text"],
},
     }
]




#Implement the two tools
def find_file(file_name):

    root = Path("D:/CST/code/Agent")

    paths=[]    
    for path in root.rglob(file_name):
        if path.is_file():
            paths.append(path)

    str_paths = [str(p) for p in paths]
    return str_paths#The API requires strings rather than Windows Path objects

def find_text(paths,text):

    locationList=[]
    if not isinstance(paths, list):
        raise TypeError(f"paths must be a list; received {type(paths).__name__}")
    #Check whether paths is a list
    pathObjs=[Path(p) for  p in paths]#Convert path strings to Path objects
   #An empty list will skip the loop 
    for path in pathObjs:
        with path.open("r",encoding="utf-8",errors="replace")as file:
            for line_number, line in enumerate(file, start=1): #Iterate over lines
               if text in line:
                   locationList.append({
                       "path":str(path),
                       "line_number":line_number,
                       }) #Record the path and line number
            
    return locationList 



#Let the model decide whether to call tools in the agent loop
inputList=[{"role":"user","content":"Find a file named '1.txt' and find the word 'lidada' if it existed"}]

maxRange=5
stop_reason=None
final_text = ""
for i in range(maxRange):
    tool_executed = False
    try:
        response = client.responses.create(
            model="deepseek-flash",
            tools=tools,
            input=inputList,
        )
    except Exception:
        logger.exception("Model request failed")
        stop_reason = "model_error"
        break

    # Store the model's tool call requests before their results.
    inputList += response.output
    tool_calls = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(item)

    for item in tool_calls:
        logger.info(
            "Tool started: step=%s tool=%s call_id=%s",
            i + 1, item.name, item.call_id,
        )
        tool_executed = True
        try:
            args = json.loads(item.arguments)
            if item.name == "find_file":
                data = find_file(args["name"])
            elif item.name == "find_text":
                data = find_text(args["paths"], args["text"])
            else:
                raise ValueError(f"Unknown Tool {item.name}")

            result = {"ok": True, "data": data}
            logger.info("Tool Calling completed")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.exception("Tool error: tool=%s", item.name)
            result = {
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

        # Return a result for each tool call, including failures.
        inputList.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result, ensure_ascii=False),
        })

    if not tool_executed:
        stop_reason = "completed"
        final_text = response.output_text
        break
else:
    stop_reason = "max_range"

print("Final status:")
if stop_reason == "completed":
    print("Completed normally")
    print("Final output:")
    print(final_text)
elif stop_reason == "model_error":
    print("Model error")
elif stop_reason == "max_range":
    print("Maximum number of iterations reached")
else:
    print("Unknown Error")

#Test cases: match found, file not found, text not found, file read failure
