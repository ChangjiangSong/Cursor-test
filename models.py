"""
模型服务模块，负责初始化和配置DeepSeek大模型
"""
from langchain_deepseek import ChatDeepSeek
from config import (
    DEEPSEEK_API_KEY, 
    DEEPSEEK_MODEL, 
    TEMPERATURE, 
    MAX_TOKENS, 
    TIMEOUT, 
    MAX_RETRIES
)

def get_llm():
    """
    初始化并返回DeepSeek LLM模型实例
    """
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
        api_key=DEEPSEEK_API_KEY
    )

# 预初始化LLM实例供全局使用
llm = get_llm() 