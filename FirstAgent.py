import os
from dotenv import load_dotenv
from openai import OpenAI
import sys

print("解释器路径:", sys.executable)
print("标准输出编码:", sys.stdout.encoding)
load_dotenv();

client=OpenAI()

response=client.responses.create(
model="deepseek-flash",
input="Say hello",
)

print(response.output_text)