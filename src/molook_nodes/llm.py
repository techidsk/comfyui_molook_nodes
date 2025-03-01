import os
from typing import Optional
from pydantic import BaseModel


class OpenAIProvider(BaseModel):
    """OpenAI API提供者配置类"""

    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def get_config(self):
        """获取配置信息，优先使用实例属性，其次使用环境变量"""
        config = {
            "base_url": self.base_url
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": self.api_key or os.getenv("OPENAI_API_KEY"),
        }

        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is required. Please provide it through input or set OPENAI_API_KEY environment variable."
            )

        return config


class OpenAIProviderNode:
    """OpenAI Provider节点类"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("OPENAI_PROVIDER",)
    RETURN_NAMES = ("provider",)

    FUNCTION = "execute"
    CATEGORY = "Molook_nodes/LLM"

    def execute(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        return (provider,)



NODE_CLASS_MAPPINGS = {
    "OpenAIProvider(Molook)": OpenAIProviderNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIProvider(Molook)": "OpenAI Provider"
}
