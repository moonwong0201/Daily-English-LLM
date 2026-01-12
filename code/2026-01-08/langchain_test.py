import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ===== 1. 环境变量 =====
# TODO: 设置 OPENAI_API_KEY 和 OPENAI_BASE_URL（通义千问兼容端点）
os.environ["OPENAI_API_KEY"] = "sk-078ae61448344f53b3cb03bcc85ff7cd"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ===== 2. 实例化 ChatOpenAI =====
# TODO: 指定模型名、temperature
llm = ChatOpenAI(
    model="qwen-turbo",
    temperature=0.9
)

# ===== 3. 提示模板 =====
# TODO: PromptTemplate 输入变量 + 中文模板字符串
prompt = PromptTemplate(
    input_variables=["product"],
    template='请为一家专门生产“{product}”的公司，想三个有创意、朗朗上口的名字。'
)

# ===== 4. 链式编排（LCEL） =====
# TODO: RunnablePassthrough → prompt → llm → 链
chain = (
    {'product': RunnablePassthrough()}
    | prompt
    | llm
)

# ===== 5. 运行 =====
# TODO: chain.invoke(输入) → 打印 response.content
product_name = "唱片"
response = chain.invoke(product_name)
print(f"为生产“{product_name}”的公司取名建议：\n")
print(response.content)

# # pip install -qU langchain-openai
# import os
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
#
# # 1. 用 ChatOpenAI 指向 DashScope 兼容端点
# os.environ["OPENAI_API_KEY"] = "sk-078ae61448344f53b3cb03bcc85ff7cd"
# os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#
# llm = ChatOpenAI(
#     model="qwen-turbo",   # 通义千问模型名
#     temperature=0.7
# )
#
# # 2. 提示模板（同原模板）
# prompt_template = PromptTemplate(
#     input_variables=["product"],
#     template="请为一家专门生产“{product}”的公司，想三个有创意、朗朗上口的名字。"
# )
#
# # 3. 链式编排（LCEL）
# # ChatOpenAI 需要消息格式，这里用 RunnablePassthrough 把字符串包成 HumanMessage
# chain = (
#     {"product": RunnablePassthrough()}  # 把输入字典转成字符串等价于lambda x: {"product": x}
#     | prompt_template                   # 生成字符串提示
#     | llm                               # ChatOpenAI 接收字符串，内部自动转成消息
# )
#
# # 4. 运行
# product_name = "个人知识管理软件"
# response = chain.invoke(product_name)
# print(f"为生产“{product_name}”的公司取名建议：\n")
# print(response.content)