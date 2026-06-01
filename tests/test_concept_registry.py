"""ConceptRegistry: IDs determinísticos, dedup por slug e migração (Passo 3)."""
import json
import os

from writer.concept_registry import ConceptRegistry, clean_concept_name, concept_id


def test_concept_id_normalizes():
    assert concept_id("Reward Shaping") == concept_id("reward  shaping")
    assert concept_id("Reward Shaping") == concept_id("REWARD SHAPING")
    assert concept_id("Estilo Agressivo!") == "concept_estilo_agressivo"


def test_clean_concept_name_strips_hallucinated_urls():
    # bug real visto rodando: o LLM colou uma URL no nome do conceito
    dirty = "Exploration vs Exploitation https://obsidian.md/notes/Exploration"
    assert clean_concept_name(dirty) == "Exploration vs Exploitation"
    assert clean_concept_name("Policy Entropy\nmais texto") == "Policy Entropy"
    assert clean_concept_name("[[Reward Shaping]]") == "Reward Shaping"
    # e o id/slug ficam limpos (sem 'httpsobsidianmd...')
    assert concept_id(dirty) == "concept_exploration_vs_exploitation"


def test_register_dedup_and_canonical(tmp_path):
    reg = ConceptRegistry(os.path.join(tmp_path, "reg.json"))
    assert reg.register("Reward Shaping", 100) is True       # inédito
    assert reg.register("reward  shaping", 200) is False      # mesmo id -> não cria
    assert reg.canonical("REWARD SHAPING") == "Reward Shaping"  # nome original mantido
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
    """Formato antigo (chaveado pelo nome) deve migrar para chaveado por id."""
    p = os.path.join(tmp_path, "reg.json")
    json.dump({"Reward Shaping": {"created_step": 5, "mentions": 2}},
              open(p, "w", encoding="utf-8"))
    reg = ConceptRegistry(p)
    assert reg.exists("reward shaping")
    assert reg.canonical("reward shaping") == "Reward Shaping"
