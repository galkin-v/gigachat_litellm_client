import uuid

from aiohttp import ClientSession, TCPConnector
from auth import GigaChatOAuthTokenAuthorizationMiddleware
from giga import GigaChatPayload, GigaChatResponse, Message
from settings import GigaSettings


class GigaChatClient:
    def __init__(self, settings: GigaSettings) -> None:
        self.url = settings.url
        self.settings = settings
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.session = ClientSession(
            connector=TCPConnector(
                limit=self.settings.limit,
                force_close=self.settings.force_close,
                ssl=False,
            ),
            middlewares=(
                GigaChatOAuthTokenAuthorizationMiddleware(
                    url=settings.auth_url,
                    access_key=settings.auth_token,
                    scope=settings.scope,
                ),
            ),
        )

    async def call(self, messages: list[Message], model: str, max_tokens: int) -> str:
        result = await self.__pure_call(
            messages=messages, headers=self.headers, model=model, max_tokens=max_tokens
        )
        return result.content

    async def __pure_call(
        self,
        messages: list[Message],
        headers: dict[str, str],
        model: str,
        max_tokens: int,
    ) -> GigaChatResponse:
        body = GigaChatPayload(
            model=model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=max_tokens,
            profanity_check=self.settings.profanity_check,
        ).model_dump_json()

        headers["x-request-id"] = str(uuid.uuid4())

        async with self.session.post(
            url=self.url,
            data=body,
            headers=headers,
            timeout=self.settings.timeout,
        ) as response:
            response.raise_for_status()

            data = GigaChatResponse(**await response.json())

            return data
