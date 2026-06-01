"""Cliente do LLM LOCAL via Ollama, com saída estruturada (JSON Schema).

Roda 100% local — sem API key, sem custo por chamada. No Ollama, structured
output é nativo: passamos `format=Model.model_json_schema()` e validamos a
resposta com Pydantic.
"""
from typing import List, Optional

from ollama import Client
from pydantic import BaseModel, Field

from writer import prompts


# ----------------------------- Schemas -----------------------------
class NewConcept(BaseModel):
    name: str = Field(description="Nome curto em Title Case, vira título de nota.")
    description: str = Field(description="1-3 frases explicando o conceito.")
    related: List[str] = Field(
        default_factory=list,
        description="Outros conceitos relacionados (nomes), para wikilinks.",
    )


class CheckpointNote(BaseModel):
    title: str = Field(description="Título curto e descritivo da nota de checkpoint.")
    headline: str = Field(description="Uma frase resumindo a mudança principal.")
    behavior_change: str = Field(
        description="Parágrafo(s) interpretando o que mudou no comportamento do agente."
    )
    evidence: List[str] = Field(
        description="Bullets com evidências numéricas específicas da janela."
    )
    linked_concepts: List[str] = Field(
        default_factory=list,
        description="Conceitos JÁ EXISTENTES a linkar (use os nomes fornecidos).",
    )
    new_concepts: List[NewConcept] = Field(
        default_factory=list,
        description="Conceitos novos a criar (somente se genuínos e reutilizáveis).",
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags do Obsidian, sem '#'."
    )


class ConceptNote(BaseModel):
    summary: str = Field(description="Explicação objetiva do conceito.")
    manifestation_in_doom: str = Field(
        description="Como o conceito aparece no treino do agente jogando Doom."
    )
    related: List[str] = Field(
        default_factory=list, description="Conceitos relacionados (wikilinks)."
    )
    tags: List[str] = Field(default_factory=list)


class RunStory(BaseModel):
    """Síntese narrativa de uma run inteira (feature A)."""

    title: str = Field(description="Título curto da síntese da run.")
    narrative: str = Field(
        description="2-4 parágrafos contando o ARCO do aprendizado, do início ao fim."
    )
    milestones: List[str] = Field(
        default_factory=list,
        description="Marcos do treino, cada um com o step aproximado em que ocorreu.",
    )
    key_concepts: List[str] = Field(
        default_factory=list, description="Conceitos centrais desta run (p/ wikilinks)."
    )


class ComparisonVerdict(BaseModel):
    """Veredito de uma comparação entre runs (feature B)."""

    summary: str = Field(description="Resumo do que difere entre as runs.")
    winner: str = Field(description="Rótulo da run vencedora, ou 'empate'.")
    reasoning: str = Field(description="Justificativa ancorada nos números fornecidos.")


# ----------------------------- Cliente -----------------------------
class LLMWriter:
    """Gera notas usando um modelo local servido pelo Ollama."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.client = Client(host=host)
        self.model = model
        # temperatura baixa -> JSON mais estável; num_ctx folgado p/ o snapshot.
        self.options = {"temperature": 0.3, "num_ctx": 8192}

    def _chat(self, system: str, user: str, schema: dict) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,          # JSON Schema -> structured output nativo
            options=self.options,
        )
        return resp.message.content

    def generate_checkpoint(
        self,
        snapshot: dict,
        previous: Optional[dict],
        existing_concepts: List[str],
        button_names: List[str],
    ) -> CheckpointNote:
        user = prompts.build_checkpoint_user_message(
            snapshot, previous, existing_concepts, button_names
        )
        content = self._chat(
            prompts.CHECKPOINT_SYSTEM, user, CheckpointNote.model_json_schema()
        )
        return CheckpointNote.model_validate_json(content)

    def generate_concept(self, name: str, hint: str) -> ConceptNote:
        user = prompts.build_concept_user_message(name, hint)
        content = self._chat(
            prompts.CONCEPT_SYSTEM, user, ConceptNote.model_json_schema()
        )
        return ConceptNote.model_validate_json(content)

    def generate_run_story(
        self, run_name: str, snapshots: List[dict], concepts: List[str]
    ) -> RunStory:
        user = prompts.build_run_story_user_message(run_name, snapshots, concepts)
        content = self._chat(
            prompts.RUNSTORY_SYSTEM, user, RunStory.model_json_schema()
        )
        return RunStory.model_validate_json(content)

    def generate_comparison(
        self, labels: List[str], summaries: dict
    ) -> ComparisonVerdict:
        user = prompts.build_comparison_user_message(labels, summaries)
        content = self._chat(
            prompts.COMPARE_SYSTEM, user, ComparisonVerdict.model_json_schema()
        )
        return ComparisonVerdict.model_validate_json(content)
