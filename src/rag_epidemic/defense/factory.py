"""Defense factory — maps a name to a ready-to-use defense instance."""

from __future__ import annotations

from typing import Any

from ..agents.verifier import EpistemicVerifier
from ..baselines.flp import FLP
from ..baselines.mask import MASK
from ..baselines.no_defense import NoDefense
from ..baselines.secon_rag import SeConRAG
from ..baselines.sem_chameleon import SemanticChameleon
from ..baselines.spark2fire import Spark2Fire
from ..defense.evp import EVP
from ..utils.openai_client import OpenAIClient


def build_defense(name: str, *, client: OpenAIClient | None = None, **kwargs: Any):
    name = name.lower()
    if name in ("undefended", "none", "no_defense"):
        return NoDefense()
    if name == "secon_rag":
        return SeConRAG(**kwargs)
    if name in ("sem_chameleon", "semantic_chameleon"):
        return SemanticChameleon(**kwargs)
    if name == "flp":
        return FLP(**kwargs)
    if name == "spark2fire":
        return Spark2Fire(**kwargs)
    if name == "mask":
        return MASK(**kwargs)
    if name == "evp":
        assert client is not None, "EVP requires an OpenAIClient (for the verifier)"
        verifier = EpistemicVerifier(client=client)
        defaults = dict(
            kstar_frac=0.10,
            gamma=0.6,
            trust_init=1.0,
            trust_threshold=0.5,
            trust_decay=0.35,
            m=2, n=3,
            quarantine_on_uncorroborated=True,
        )
        defaults.update(kwargs)
        return EVP(verifier=verifier, **defaults)
    raise ValueError(f"Unknown defense: {name}")


ALL_DEFENSES = ["undefended", "secon_rag", "sem_chameleon", "flp",
                "spark2fire", "mask", "evp"]
