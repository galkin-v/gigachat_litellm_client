from pydantic_settings import BaseSettings, SettingsConfigDict


class GigaSettings(BaseSettings):
    url: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    limit: int = 100
    force_close: bool = True
    timeout: int = 3

    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    profanity_check: bool = False
    temperature: float = 0.0000001
    scope: str = "GIGACHAT_API_CORP"
    verify_ssl_certs: bool = False
    use_kv: bool = True
    auth_token: str = ""
    json_parser: str = r"json\s*([\s\S]*?)\s*"
    message_history_limit: int = 8

    model_config = SettingsConfigDict(
        env_prefix="GIGA__",
        env_nested_max_split=1,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
