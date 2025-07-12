# models/llm_client.py

import os
import logging
from typing import Dict
# 이제 두 종류의 클라이언트를 모두 임포트합니다.
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

class LLMClient:
    """
    설정에 따라 표준 OpenAI 또는 Azure OpenAI 클라이언트를 생성하고,
    통신을 중계하는 폴리모픽(polymorphic) 클라이언트.
    """
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.provider = self.config.get("provider", "openai") # 기본값은 openai
        
        # 설정된 provider에 따라 적절한 클라이언트 생성 메서드를 호출합니다.
        if self.provider == "openai":
            self.client = self._create_openai_client()
        elif self.provider == "azure_openai":
            self.client = self._create_azure_openai_client()
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자입니다: {self.provider}")

    def _create_openai_client(self) -> OpenAI:
        """표준 OpenAI 클라이언트를 생성합니다."""
        self.logger.info("표준 OpenAI 클라이언트를 초기화합니다...")
        try:
            api_key = os.environ['OPENAI_API_KEY']
            client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI 클라이언트가 성공적으로 초기화되었습니다.")
            return client
        except KeyError:
            self.logger.error("필수 환경 변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")
            raise

    def _create_azure_openai_client(self) -> AzureOpenAI:
        """Azure OpenAI 클라이언트를 생성합니다."""
        self.logger.info("Azure OpenAI 클라이언트를 초기화합니다...")
        try:
            endpoint = os.environ['AZURE_ENDPOINT_URL']
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=self.config.get("azure_openai", {}).get("api_version", "2024-02-01")
            )
            self.logger.info("AzureOpenAI 클라이언트가 성공적으로 초기화되었습니다.")
            return client
        except KeyError as e:
            self.logger.error(f"필수 환경 변수가 설정되지 않았습니다: {e}")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """주어진 프롬프트를 바탕으로 LLM 응답을 생성합니다."""
        
        # Provider에 따라 사용할 모델/배포 이름을 가져옵니다.
        if self.provider == "openai":
            model_identifier = self.config.get("openai", {}).get("model")
        elif self.provider == "azure_openai":
            model_identifier = self.config.get("azure_openai", {}).get("deployment_name")
        else: # Should not happen due to __init__ check
            return ""

        if not model_identifier:
            self.logger.error(f"'{self.provider}'에 대한 model 또는 deployment_name이 설정되지 않았습니다.")
            return ""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        generation_params = self.config.get("generation_params", {})
        
        self.logger.info(f"'{model_identifier}' 모델({self.provider})에 요청을 보냅니다...")
        try:
            # model 파라미터에 provider에 맞는 식별자를 전달합니다.
            completion = self.client.chat.completions.create(
                model=model_identifier,
                messages=messages,
                **generation_params
            )
            response_content = completion.choices[0].message.content
            self.logger.info("LLM으로부터 성공적으로 응답을 받았습니다.")
            return response_content.strip() if response_content else ""
        except Exception as e:
            self.logger.error(f"LLM API 호출 중 오류 발생: {e}")
            return ""
