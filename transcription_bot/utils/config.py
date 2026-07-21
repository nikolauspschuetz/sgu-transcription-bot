import importlib.resources as pkg_resources
from pathlib import Path
from typing import Any, Protocol, cast

from dynaconf import Dynaconf, Validator
from dynaconf.validator import ValidatorList

# Internal data paths
DATA_FOLDER = Path(str(pkg_resources.files("transcription_bot").joinpath("data")))
VOICEPRINT_FILE = DATA_FOLDER / "voiceprint_map.json"
TEMPLATES_FOLDER = DATA_FOLDER / "templates"
CONFIG_FILE = DATA_FOLDER / "config.toml"

# Episodes that will raise exceptions when processed
UNPROCESSABLE_EPISODES = {
    # No lyrics - episodes 1-208 do not have embedded lyrics
    *range(1, 208 + 1),
    # Episodes that we cannot process
    300,  # Missing news item text
    320,  # News item #3 has unexpected line break
    502,  # News items contains a non-standard item
    875,  # Issue with news items identified as SOF
}


# Credentials that are only required when a specific backend is selected. This keeps
# free/local runs from demanding paid-service keys. Each entry: (var, when-condition).
_BACKEND_REQUIRED_ENV_VARS = [
    # Paid ASR
    ("azure_subscription_key", Validator("transcription_backend", eq="azure")),
    ("azure_service_region", Validator("transcription_backend", eq="azure")),
    # Paid diarization
    ("pyannote_token", Validator("diarization_backend", eq="pyannote_ai")),
    ("ngrok_token", Validator("diarization_backend", eq="pyannote_ai")),
    # Free local diarization needs a (free) HuggingFace token to pull gated models
    ("hf_token", Validator("diarization_backend", eq="local_pyannote")),
    # Paid LLM
    ("openai_organization", Validator("llm_backend", eq="openai")),
    ("openai_project", Validator("llm_backend", eq="openai")),
    ("openai_api_key", Validator("llm_backend", eq="openai")),
]

# Wiki credentials are only needed to publish (prod). "Output first" dev runs skip them.
_PROD_ONLY_ENV_VARS = ["wiki_username", "wiki_password", "sentry_dsn", "cronitor_api_key", "cronitor_job_id"]


class ConfigProto(Protocol):
    """Protocol for config object."""

    # Built-ins
    validators: ValidatorList

    def load_file(  # noqa: D102
        self,
        path: str | Path | None = None,
        env: str | None = None,
        silent: bool = True,  # noqa: FBT001, FBT002
        key: str | None = None,
        validate: Any = None,
    ) -> None: ...

    # Config variables
    local_mode: bool

    # RSS feeds
    podcast_rss_url: str
    wiki_rss_url: str

    # Backend selection
    transcription_backend: str
    diarization_backend: str
    llm_backend: str

    # Local Whisper (faster-whisper)
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_beam_size: int

    # Local diarization (pyannote.audio)
    hf_token: str
    diarization_pipeline: str
    diarization_device: str
    voiceprint_dir: str

    # Local LLM (ollama)
    ollama_base_url: str
    ollama_model: str

    # Wiki
    wiki_username: str
    wiki_password: str
    wiki_episode_url_base: str
    wiki_api_base: str

    # Azure / transcription
    azure_subscription_key: str
    azure_service_region: str

    # pyannote / diarization
    pyannote_token: str
    pyannote_identify_endpoint: str
    pyannote_voiceprint_endpoint: str
    pyannote_jobs_endpoint: str

    # Local server
    ngrok_token: str
    server_port: int

    # OpenAI / GPT / llm
    openai_organization: str
    openai_project: str
    openai_api_key: str
    llm_model: str

    # Monitoring (only in deployed)
    sentry_dsn: str
    cronitor_api_key: str
    cronitor_job_id: str


_prod_only_validators = [
    Validator(
        var_name,
        required=True,
        ne="",
        messages={"operations": "{name} must not be blank when in production"},
        when=Validator("local_mode", eq=False),
    )
    for var_name in _PROD_ONLY_ENV_VARS
]

_backend_validators = [
    Validator(
        var_name,
        required=True,
        ne="",
        messages={"operations": "{name} must not be blank for the selected backend"},
        when=when_condition,
    )
    for var_name, when_condition in _BACKEND_REQUIRED_ENV_VARS
]

config = Dynaconf(
    envvar_prefix="TB",
    settings_files=[CONFIG_FILE],
    load_dotenv=True,
    ignore_unknown_envvars=True,
    validators=[
        Validator("log_level", cast=lambda x: x.upper()),
        Validator("local_mode", cast=bool),
    ],
)

config = cast("ConfigProto", config)

config.validators.register(
    *_prod_only_validators,
    *_backend_validators,
)
