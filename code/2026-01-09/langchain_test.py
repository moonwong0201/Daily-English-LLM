import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 用 ChatOpenAI 指向 DashScope 兼容端点
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

llm = ChatOpenAI(
    model="qwen-turbo",   # 通义千问模型名
    temperature=0.7
)

# 2. 提示模板（同原模板）
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="请为一家专门生产“{product}”的公司，想三个有创意、朗朗上口的名字。"
)

# 3. 链式编排（LCEL）
# ChatOpenAI 需要消息格式，这里用 RunnablePassthrough 把字符串包成 HumanMessage
chain = (
    {"product": RunnablePassthrough()}  # 把输入字典转成字符串等价于lambda x: {"product": x}
    | prompt_template                   # 生成字符串提示
    | llm                               # ChatOpenAI 接收字符串，内部自动转成消息
)

# 4. 运行
product_name = "唱片"
response = chain.invoke(product_name)
print(f"为生产“{product_name}”的公司取名建议：\n")
print(response.content)
