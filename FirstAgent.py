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
    print(str_paths)
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



#实现两个工具的调用，一共需要两次iteration（response）


inputList=[{"role":"user","content":"Find a file named '1.txt' and find the word 'lidada' if it existed"}]

response=client.responses.create(

model="deepseek-flash",
tools=[tools[0]],#接口传的是列表
input=inputList,

)


inputList += response.output #存储模型的tool calling 请求

#存储tool calling output
for item in response.output:
    if item.type=="function_call":#function_call 而不是function
        if item.name=="find_file":
          #执行函数
          name=json.loads(item.arguments)["name"]
          paths=find_file(name)
         #将tool call output打包加入context(json)
          inputList.append(
                 {
                    "type":"function_call_output",
                    "call_id":item.call_id,
                    "output": json.dumps(paths, ensure_ascii=False),
                 }
          ) 



response=client.responses.create(
model="deepseek-flash",
tools=[tools[1]],
input=inputList,

)

inputList += response.output #存储模型的tool calling 请求

#存储tool calling output
for item in response.output:
    if item.type=="function_call":
        if item.name=="find_text":
          #执行函数
          

          paths=json.loads(item.arguments)["paths"]
          text=json.loads(item.arguments)["text"]

          location=find_text(paths,text)
         #将tool call output打包加入context(json)
          inputList.append(
                 {
                    "type":"function_call_output",
                    "call_id":item.call_id,
                    "output": json.dumps(location, ensure_ascii=False),#这里json格式的数组会在传输时自动变成符合传输协议的字符串
                 }
          ) 

response=client.responses.create(
    model="deepseek-flash",
    input=inputList,
    instructions="Summerize all locations of specific text as a list,if there's nothing location then send 'fail to find'",

)

print("Final output:")
print(response.output_text)