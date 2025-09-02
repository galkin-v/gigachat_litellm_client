## Example usage

```python
from gigachat_litellm_provider import gigachat_handler

response = gigachat_handler.completion(
    "GigaChat-2",
    messages=[{"role": "user", "content": "Привет"}]
)

print(response.content)  # 'Привет. Как настроение?'
```

## Example usage in Jupyter Notebook

```python
import nest_asyncio
from gigachat_litellm_provider import gigachat_handler

nest_asyncio.apply()

response = gigachat_handler.completion(
    "GigaChat-2",
    messages=[{"role": "user", "content": "Привет"}]
)

print(response.content)  # 'Привет. Как настроение?'
```