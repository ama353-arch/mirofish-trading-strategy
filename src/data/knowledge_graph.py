"""
Knowledge Graph — a simplified GraphRAG implementation inspired by MiroFish.

Uses NetworkX locally (no external graph DB dependency) with LLM-based entity
and relationship extraction from text documents.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    entity_type: str          # Person, Organization, Event, Policy, Instrument, etc.
    summary: str = ""
    attributes: dict = field(default_factory=dict)
    source_documents: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str        # INFLUENCES, CAUSES, CORRELATES, OPPOSES, etc.
    description: str = ""
    strength: float = 0.5     # 0-1
    temporal: str = ""        # When this relationship is relevant
    source_documents: list[str] = field(default_factory=list)


class KnowledgeGraph:
    """
    Local knowledge graph built from text documents using LLM extraction.

    Inspired by MiroFish's GraphRAG pipeline:
    1. Ingest raw text documents
    2. Extract entities and relationships via LLM
    3. Build a NetworkX graph
    4. Query for context relevant to simulation
    """

    # Entity types relevant to market prediction
    ENTITY_TYPES = [
        "Person", "Organization", "Country", "Policy",
        "FinancialInstrument", "EconomicIndicator",
        "Event", "Sector", "Technology", "PoliticalParty",
    ]

    RELATION_TYPES = [
        "INFLUENCES", "CAUSES", "OPPOSES", "SUPPORTS",
        "CORRELATES_WITH", "LEADS", "REGULATES",
        "INVESTS_IN", "COMPETES_WITH", "PART_OF",
    ]

    def __init__(self, llm_client=None):
        self.graph = nx.DiGraph()
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.documents: dict[str, str] = {}  # doc_id -> text
        self.llm = llm_client

    def add_document(self, text: str, source_id: str = "") -> str:
        """Add a raw text document to the graph. Returns document ID."""
        doc_id = source_id or hashlib.md5(text[:500].encode()).hexdigest()[:12]
        self.documents[doc_id] = text
        logger.info(f"Added document {doc_id} ({len(text)} chars)")
        return doc_id

    def build_from_documents(self, use_llm: bool = True) -> dict:
        """
        Process all ingested documents to extract entities and relationships.

        Returns stats about the built graph.
        """
        if use_llm and self.llm:
            return self._build_with_llm()
        else:
            return self._build_rule_based()

    def _build_with_llm(self) -> dict:
        """Extract entities and relationships using LLM."""
        all_entities = []
        all_relationships = []

        for doc_id, text in self.documents.items():
            # Chunk long documents
            chunks = self._chunk_text(text, chunk_size=2000, overlap=200)
            for chunk in chunks:
                entities, rels = self._extract_from_chunk(chunk, doc_id)
                all_entities.extend(entities)
                all_relationships.extend(rels)

        # Deduplicate entities by name (case-insensitive)
        seen = {}
        for ent in all_entities:
            key = ent.name.lower().strip()
            if key not in seen:
                seen[key] = ent
                self.entities[ent.id] = ent
                self.graph.add_node(ent.id, **asdict(ent))
            else:
                # Merge source documents
                seen[key].source_documents.extend(ent.source_documents)

        # Add relationships
        for rel in all_relationships:
            if rel.source_id in self.entities and rel.target_id in self.entities:
                self.relationships.append(rel)
                self.graph.add_edge(
                    rel.source_id, rel.target_id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    strength=rel.strength,
                )

        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents_processed": len(self.documents),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
        }
        logger.info(f"Knowledge graph built: {stats}")
        return stats

    def _extract_from_chunk(self, text: str, doc_id: str) -> tuple[list[Entity], list[Relationship]]:
        """Use LLM to extract entities and relationships from a text chunk."""
        system = f"""You are an entity and relationship extractor for a financial knowledge graph.
Extract entities and relationships from the given text.

Entity types: {', '.join(self.ENTITY_TYPES)}
Relationship types: {', '.join(self.RELATION_TYPES)}

Respond in JSON with:
{{
  "entities": [
    {{"name": "...", "entity_type": "...", "summary": "..."}}
  ],
  "relationships": [
    {{"source": "entity_name", "target": "entity_name", "relation_type": "...", "description": "..."}}
  ]
}}

Extract only clearly stated entities and relationships. Do not hallucinate."""

        try:
            result = self.llm.chat_json(system, f"Extract from:\n\n{text}", temperature=0.3)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return [], []

        entities = []
        name_to_id = {}
        for e in result.get("entities", []):
            eid = hashlib.md5(e["name"].lower().encode()).hexdigest()[:10]
            name_to_id[e["name"]] = eid
            entities.append(Entity(
                id=eid,
                name=e["name"],
                entity_type=e.get("entity_type", "Person"),
                summary=e.get("summary", ""),
                source_documents=[doc_id],
            ))

        relationships = []
        for r in result.get("relationships", []):
            src = name_to_id.get(r.get("source"))
            tgt = name_to_id.get(r.get("target"))
            if src and tgt:
                relationships.append(Relationship(
                    source_id=src,
                    target_id=tgt,
                    relation_type=r.get("relation_type", "INFLUENCES"),
                    description=r.get("description", ""),
                    source_documents=[doc_id],
                ))

        return entities, relationships

    def _build_rule_based(self) -> dict:
        """Simple keyword-based entity extraction (no LLM needed)."""
        import re

        # Very simple NER: extract capitalized multi-word phrases
        for doc_id, text in self.documents.items():
            # Find capitalized phrases (rough NER)
            pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
            names = set(re.findall(pattern, text))
            for name in names:
                eid = hashlib.md5(name.lower().encode()).hexdigest()[:10]
                if eid not in self.entities:
                    entity = Entity(
                        id=eid, name=name,
                        entity_type="Person" if len(name.split()) <= 3 else "Organization",
                        summary=f"Entity mentioned in documents",
                        source_documents=[doc_id],
                    )
                    self.entities[eid] = entity
                    self.graph.add_node(eid, **asdict(entity))

        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents_processed": len(self.documents),
        }
        logger.info(f"Rule-based graph built: {stats}")
        return stats

    def get_context_for_event(self, event_description: str, max_entities: int = 20) -> str:
        """
        Query the knowledge graph for context relevant to a specific event.
        Returns a text summary suitable for agent consumption.
        """
        if not self.entities:
            return "No knowledge graph context available."

        # Simple keyword matching (upgrade to embeddings for production)
        event_words = set(event_description.lower().split())
        scored_entities = []
        for eid, entity in self.entities.items():
            entity_words = set(entity.name.lower().split()) | set(entity.summary.lower().split())
            overlap = len(event_words & entity_words)
            if overlap > 0:
                scored_entities.append((overlap, entity))

        scored_entities.sort(key=lambda x: x[0], reverse=True)
        top = scored_entities[:max_entities]

        if not top:
            # Fall back to most connected nodes
            centrality = nx.degree_centrality(self.graph)
            top_ids = sorted(centrality, key=centrality.get, reverse=True)[:max_entities]
            top = [(0, self.entities[eid]) for eid in top_ids if eid in self.entities]

        lines = ["=== Knowledge Graph Context ===\n"]
        for _, entity in top:
            lines.append(f"• {entity.name} ({entity.entity_type}): {entity.summary}")
            # Add relationships
            for rel in self.relationships:
                if rel.source_id == entity.id:
                    target = self.entities.get(rel.target_id)
                    if target:
                        lines.append(f"  → {rel.relation_type} {target.name}: {rel.description}")

        return "\n".join(lines)

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_stats(self) -> dict:
        """Return graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": dict(pd.Series([e.entity_type for e in self.entities.values()]).value_counts())
                if self.entities else {},
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
        }

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def to_dict(self) -> dict:
        """Serialize the graph."""
        return {
            "entities": {eid: asdict(e) for eid, e in self.entities.items()},
            "relationships": [asdict(r) for r in self.relationships],
            "documents": list(self.documents.keys()),
        }
