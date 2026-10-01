import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv();

client=OpenAI()

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
    "required":["name"],#声明用required,与properties是同级的
    },

    },
     {
    "type":"function",
    "description":"If there are one or more file paths,Find the specific text location from the path motioned by customer",
    "name":"find_text",

    "parameters":{
        "type":"object",
        "properties":{
            "paths":{#paths是列表，不是一个字符串
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




#实现两个工具
def find_file(file_name):

    root = Path("D:/CST/code/Agent")

    paths=[]    
    for path in root.rglob(file_name):
        if path.is_file():
            paths.append(path)

    str_paths = [str(p) for p in paths]
    return str_paths#接口需要接收字符串而不是windows路径

def find_text(paths,text):

    locationList=[]
    if not isinstance(paths, list):
        raise TypeError(f"paths 应为列表，实际为 {type(paths).__name__}")
    #检验是否为列表
    pathObjs=[Path(p) for  p in paths]#pathStr转化为pathObj
   #同时如果这里是空列表则会报错 
    for path in pathObjs:
        with path.open("r",encoding="utf-8",errors="replace")as file:
            for line_number, line in enumerate(file, start=1): #按行迭代
               if text in line:
                   locationList.append({
                       "path":str(path),
                       "line_number":line_number,
                       }) #记录路径和行号
            
    return locationList 



#让model自动推断是否需要调用工具以实现agent loop
inputList=[{"role":"user","content":"Find a file named '1.txt' and find the word 'lidada' if it existed"}]

maxRange=5
for i in range(maxRange):

    tool_executed=False

    response=client.responses.create(

     model="deepseek-flash",
     tools=tools,#接口传的是列表
     input=inputList,

    )

    inputList += response.output #存储模型的tool calling 请求

#存储tool calling output
    for item in response.output:
      if item.type=="function_call":#function_call 而不是function
        
         tool_executed=True

         args=json.loads(item.arguments)#json.load读取对象,json.loads读取字符串
         if item.name=="find_file":
          #执行函数
            result=find_file(args["name"])
         #将tool call output打包加入context(json)
         elif item.name=="find_text":
            result=find_text(args["paths"],args["text"])
         #将tool call output打包加入context(json)
         else:
             raise ValueError(f"Unknown Tool {item.name}")
         inputList.append(
              {
                    "type":"function_call_output",
                    "call_id":item.call_id,
                     "output": json.dumps(result, ensure_ascii=False),#这里json格式的数组会在传输时自动变成符合传输协议的字符串
               }
           ) 



    if tool_executed==False:
         break



response=client.responses.create(
    model="deepseek-flash",
    input=inputList,
    instructions="Summerize all locations of specific text as a list,if there's nothing location then send 'fail to find'",

)

print("Final output:")
print(response.output_text)