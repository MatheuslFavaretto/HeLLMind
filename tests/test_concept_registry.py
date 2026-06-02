"""ConceptRegistry: deterministic IDs, slug dedup, and migration."""
import json
import os

from writer.concept_registry import ConceptRegistry, clean_concept_name, concept_id


def test_concept_id_normalizes():
    assert concept_id("Reward Shaping") == concept_id("reward  shaping")
    assert concept_id("Reward Shaping") == concept_id("REWARD SHAPING")
    assert concept_id("Estilo Agressivo!") == "concept_estilo_agressivo"


def test_clean_concept_name_strips_hallucinated_urls():
    # real bug seen at runtime: the LLM glued a URL onto the concept name
    dirty = "Exploration vs Exploitation https://obsidian.md/notes/Exploration"
    assert clean_concept_name(dirty) == "Exploration vs Exploitation"
    assert clean_concept_name("Policy Entropy\nmais texto") == "Policy Entropy"
    assert clean_concept_name("[[Reward Shaping]]") == "Reward Shaping"
    # e o id/slug ficam limpos (sem 'httpsobsidianmd...')
    assert concept_id(dirty) == "concept_exploration_vs_exploitation"


def test_clean_concept_name_strips_trend_and_value_tails():
    # real bug seen at 150k: the 3b appended trends/values to concept names
    assert clean_concept_name("Action Entropy down from 09 to 07") == "Action Entropy"
    assert clean_concept_name("Action Entropy Down") == "Action Entropy"
    assert clean_concept_name("Accuracy 25") == "Accuracy"
    # all collapse to the same id -> one note
    cid = concept_id("Action Entropy")
    assert concept_id("Action Entropy Down") == cid
    assert concept_id("Action Entropy down from 09 to 07") == cid
    # legit names are untouched
    assert clean_concept_name("Sample Efficiency") == "Sample Efficiency"
    assert clean_concept_name("Exploration vs Exploitation") == "Exploration vs Exploitation"


def test_register_dedup_and_canonical(tmp_path):
    reg = ConceptRegistry(os.path.join(tmp_path, "reg.json"))
    assert reg.register("Reward Shaping", 100) is True       # inédito
    assert reg.register("reward  shaping", 200) is False      # same id -> not created
    assert reg.canonical("REWARD SHAPING") == "Reward Shaping"  # original name kept
    assert reg.exists("reward shaping") is True
    assert reg.names() == ["Reward Shaping"]


def test_touch_increments_mentions(tmp_path):
    p = os.path.join(tmp_path, "reg.json")
    reg = ConceptRegistry(p)
    reg.register("Policy Entropy", 1)
    reg.touch("policy entropy")
    reg.touch("POLICY ENTROPY")
    data = json.load(open(p, encoding="utf-8"))
    assert data[concept_id("Policy Entropy")]["mentions"] == 3


def test_migration_from_name_keyed(tmp_path):
    """Old (name-keyed) format must migrate to the id-keyed format."""
    p = os.path.join(tmp_path, "reg.json")
    json.dump({"Reward Shaping": {"created_step": 5, "mentions": 2}},
              open(p, "w", encoding="utf-8"))
    reg = ConceptRegistry(p)
    assert reg.exists("reward shaping")
    assert reg.canonical("reward shaping") == "Reward Shaping"
