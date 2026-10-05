"""Project exception hierarchy."""


class LaborCodeRagError(Exception):
    """Base class for all project errors."""


class RetrievalError(LaborCodeRagError):
    """Retrieval (vector, BM25, fusion, rerank) failed."""


class LLMError(LaborCodeRagError):
    """An LLM or embedding call failed."""


class SourceDownloadError(LaborCodeRagError):
    """The official law text could not be downloaded."""


class ParseError(LaborCodeRagError):
    """The law text could not be parsed into articles."""
