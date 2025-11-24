from typing import List

try:
    from deepseek_tokenizer import DeepseekV3Tokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency required
    raise ImportError("DeepSeek v3 tokenizer package is required for this project.") from exc


class DeepSeekTokenizer:
    def __init__(self):
        self._tok = DeepseekV3Tokenizer()

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def batch_encode(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]


tokenizer = DeepSeekTokenizer()
