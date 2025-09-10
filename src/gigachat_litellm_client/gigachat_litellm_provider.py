import asyncio
import uuid
from typing import Dict, List, Optional

import aiohttp
from litellm import APIError, Choices, CustomLLM, ModelResponse, RateLimitError
from litellm.utils import Message as LiteLLMMessage
from litellm.utils import Usage

from .auth import GigaChatOAuthTokenAuthorizationMiddleware
from .giga import GigaChatPayload, GigaChatResponse
from .settings import GigaSettings


class CustomModelResponse(ModelResponse):
    @property
    def content(self) -> str:
        return self.choices[0].message.content


class GigaChatLLM(CustomLLM):
    def __init__(self):
        super().__init__()
        try:
            self.settings = GigaSettings()
        except Exception as e:
            raise ValueError(
                f"Failed to load GigaChat settings. Ensure GIGA_* env variables are set. Error: {e}"
            )
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        try:
            if self._session is not None and not self._session.closed:
                return self._session
            
            if self._session is not None:
                await self._session.close()
            
            connector = aiohttp.TCPConnector(
                limit=self.settings.limit,
                force_close=self.settings.force_close,
                ssl=self.settings.verify_ssl_certs,
                enable_cleanup_closed=True,
            )
            auth_middleware = GigaChatOAuthTokenAuthorizationMiddleware(
                url=self.settings.auth_url,
                access_key=self.settings.auth_token,
                scope=self.settings.scope,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                middlewares=[auth_middleware],
                timeout=aiohttp.ClientTimeout(
                    total=self.settings.timeout,
                    connect=self.settings.timeout,
                    sock_connect=self.settings.timeout,
                    sock_read=self.settings.timeout,
                ),
            )
            return self._session
        except Exception as e:
            if self._session:
                await self._session.close()
            raise APIError(
                message=f"Failed to create session: {str(e)}",
                llm_provider="gigachat-provider",
                status_code=500,
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure the session is closed gracefully on exit."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def acompletion(
        self, model: str, messages: List[Dict | LiteLLMMessage], **kwargs
    ) -> CustomModelResponse:
        """The main async completion method called by LiteLLM."""
        session = await self._get_session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-request-id": str(uuid.uuid4()),
        }

        if isinstance(messages[0], dict):
            messages = [
                LiteLLMMessage(role=msg["role"], content=str(msg["content"]))
                for msg in messages
            ]

        actual_model_name = model.split("/")[-1]

        optional_params = kwargs.get("optional_params", {})

        messages_dicts = [
            m.json() if isinstance(m, LiteLLMMessage) else m for m in messages
        ]

        payload_data = {
            "model": actual_model_name,
            "messages": messages_dicts,
            "temperature": optional_params.get("temperature", 1.0),
            **{
                k: v
                for k, v in optional_params.items()
                if k in GigaChatPayload.model_fields
            },
        }
        payload = GigaChatPayload(**payload_data, max_tokens=512, profanity_check=False)

        try:
            async with session.post(
                url=self.settings.url,
                data=payload.model_dump_json(),
                headers=headers,
            ) as response:
                if response.status == 429:
                    raise RateLimitError(
                        message="GigaChat API rate limit exceeded.",
                        llm_provider="gigachat-provider",
                        model=model,
                    )
                response.raise_for_status()
                response_json = await response.json()
                giga_response = GigaChatResponse(**response_json)

            return self.convert_response(giga_response, original_model=model)

        except aiohttp.ClientResponseError as e:
            raise APIError(
                message=f"GigaChat API Error: {e.status} {e.message}",
                llm_provider="gigachat-provider",
                model=model,
                status_code=e.status,
            )
        except Exception as e:
            raise APIError(
                message=str(e),
                llm_provider="gigachat-provider",
                model=model,
                status_code=500,
            )

    def completion(
        self, model: str, messages: List[Dict], **kwargs
    ) -> CustomModelResponse:
        """Synchronous wrapper for acompletion."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def _run_completion():
            try:
                return await self.acompletion(model=model, messages=messages, **kwargs)
            finally:
                if self._session and not self._session.closed:
                    await self._session.close()
        
        return loop.run_until_complete(_run_completion())

    def convert_response(
        self, giga_response: GigaChatResponse, original_model: str
    ) -> CustomModelResponse:
        """Helper to convert GigaChat's response to LiteLLM's standard CustomModelResponse."""
        model_response = CustomModelResponse(
            id=str(uuid.uuid4()),
            choices=[
                Choices(
                    finish_reason=choice.finish_reason,
                    index=choice.index,
                    message=LiteLLMMessage(
                        content=choice.message.content, role=choice.message.role
                    ),
                )
                for choice in giga_response.choices
            ],
            model=original_model,
            usage=Usage(
                prompt_tokens=giga_response.usage.prompt_tokens,
                completion_tokens=giga_response.usage.completion_tokens,
                total_tokens=giga_response.usage.total_tokens,
            ),
        )
        model_response._hidden_params["original_response"] = giga_response.model_dump()
        return model_response


gigachat_handler = GigaChatLLM()
