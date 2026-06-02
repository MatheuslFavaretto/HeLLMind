"""Local LLM client via Ollama, with structured output (JSON Schema).

Runs 100% locally — no API key, no per-call cost. In Ollama, structured output is
native: we pass `format=Model.model_json_schema()` and validate the response with
Pydantic.
"""
from typing import List, Optional

from ollama import Client
from pydantic import BaseModel, Field

from writer import prompts


# ----------------------------- Schemas -----------------------------
class NewConcept(BaseModel):
    name: str = Field(description="Short Title Case name; becomes a note title.")
    description: str = Field(description="1-3 sentences explaining the concept.")
    related: List[str] = Field(
        default_factory=list,
        description="Other related concepts (names), for wikilinks.",
    )


class CheckpointNote(BaseModel):
    title: str = Field(description="Short, descriptive title for the checkpoint note.")
    headline: str = Field(description="One sentence summarizing the main change.")
    behavior_change: str = Field(
        description="Paragraph(s) interpreting what changed in the agent's behavior."
    )
    evidence: List[str] = Field(
        description="Bullets with specific numeric evidence from the window."
    )
    linked_concepts: List[str] = Field(
        default_factory=list,
        description="EXISTING concepts to link (use the provided names).",
    )
    new_concepts: List[NewConcept] = Field(
        default_factory=list,
        description="New concepts to create (only if genuine and reusable).",
    )
    tags: List[str] = Field(
        default_factory=list, description="Obsidian tags, without '#'."
    )


class ConceptNote(BaseModel):
    summary: str = Field(description="Objective explanation of the concept.")
    manifestation_in_doom: str = Field(
        description="How the concept shows up while training the agent on Doom."
    )
    related: List[str] = Field(
        default_factory=list, description="Related concepts (wikilinks)."
    )
    tags: List[str] = Field(default_factory=list)


class RunStory(BaseModel):
    """Narrative synthesis of a whole run (feature A)."""

    title: str = Field(description="Short title for the run synthesis.")
    narrative: str = Field(
        description="2-4 paragraphs telling the ARC of the learning, start to finish."
    )
    milestones: List[str] = Field(
        default_factory=list,
        description="Milestones, each citing the approximate step where it happened.",
    )
    key_concepts: List[str] = Field(
        default_factory=list, description="Core concepts of this run (for wikilinks)."
    )


class ComparisonVerdict(BaseModel):
    """Verdict of a comparison between runs (feature B)."""

    summary: str = Field(description="Summary of what differs between the runs.")
    winner: str = Field(description="Label of the winning run, or 'tie'.")
    reasoning: str = Field(description="Justification anchored in the given numbers.")


class Lesson(BaseModel):
    """A single reusable lesson learned across runs (Phase 4)."""

    title: str = Field(description="Short title for the lesson.")
    insight: str = Field(description="The actionable lesson, e.g. a failure pattern.")
    evidence: str = Field(description="The numbers from the report that support it.")


class LessonsNote(BaseModel):
    lessons: List[Lesson] = Field(
        default_factory=list,
        description="3-6 concrete lessons grounded in the aggregated numbers.",
    )


class RewardTweak(BaseModel):
    """A proposed change to one reward-shaping weight (Phase 6)."""

    knob: str = Field(description="One of: hit_reward, miss_penalty, "
                                 "damage_taken_penalty, death_penalty.")
    suggested: float = Field(description="Proposed new value for this knob.")
    reason: str = Field(description="Why, anchored in the observed numbers.")


class RewardSuggestions(BaseModel):
    summary: str = Field(description="One-line summary of the behavior issue seen.")
    tweaks: List[RewardTweak] = Field(default_factory=list)


# ----------------------------- Client -----------------------------
class LLMWriter:
    """Generates notes using a local model served by Ollama."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 4096,
        num_predict: int = 700,
        keep_alive: str = "5m",
        timeout: float = 300.0,
    ) -> None:
        # A read timeout is ESSENTIAL: without it a lost/stalled Ollama response
        # blocks the socket recv forever (the whole post-train notes phase hangs).
        self.client = Client(host=host, timeout=timeout)
        self.model = model
        self.keep_alive = keep_alive
        # low temperature -> stable JSON; small num_ctx (fact sheet is tiny) saves
        # memory; num_predict caps generation -> faster and bounded.
        self.options = {
            "temperature": 0.3,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

    def _chat(self, system: str, user: str, schema: dict) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema,          # JSON Schema -> native structured output
            options=self.options,
            keep_alive=self.keep_alive,  # keep the model warm across the batch
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

    def generate_lessons(self, stats: dict) -> LessonsNote:
        user = prompts.build_lessons_user_message(stats)
        content = self._chat(
            prompts.LESSONS_SYSTEM, user, LessonsNote.model_json_schema()
        )
        return LessonsNote.model_validate_json(content)

    def generate_reward_suggestions(
        self, stats: dict, weights: dict
    ) -> RewardSuggestions:
        user = prompts.build_suggest_user_message(stats, weights)
        content = self._chat(
            prompts.SUGGEST_SYSTEM, user, RewardSuggestions.model_json_schema()
        )
        return RewardSuggestions.model_validate_json(content)
