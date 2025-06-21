from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class ConnectionType(str, Enum):
    GENERAL_SPECIFIC = "general_specific"
    SPECIFIC_GENERAL = "specific_general"
    CAUSE_EFFECT = "cause_effect"
    EFFECT_CAUSE = "effect_cause"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SIMILARITY = "similarity"
    CONTRAST = "contrast"
    CONTEXT = "context"

class Connection(BaseModel):
    target_node_id: str
    connection_type: ConnectionType
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)

class MemoryNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    concept: str = Field(..., description="Main concept or idea")
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Brief summary of the memory")
    full_content: str = Field(..., description="Complete memory content")
    connections: Dict[str, Connection] = Field(default_factory=dict)
    
    # Metadata
    access_count: int = Field(default=0, ge=0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Vector reference
    embedding_id: Optional[str] = None
    
    class Config:
        pass
    
    @property
    def key(self) -> str:
        '''Generate the key from concept + keywords'''
        keywords_str = ", ".join(self.keywords) if self.keywords else ""
        return f"{self.concept}: {keywords_str}" if keywords_str else self.concept
    
    def add_connection(self, target_id: str, connection_type: ConnectionType, initial_weight: float = 0.0):
        '''Add a new connection to another node'''
        connection = Connection(
            target_node_id=target_id,
            connection_type=connection_type,
            weight=initial_weight
        )
        self.connections[target_id] = connection
    
    def strengthen_connection(self, target_id: str, strength_increment: float = 0.1):
        '''Strengthen an existing connection'''
        if target_id in self.connections:
            connection = self.connections[target_id]
            connection.weight = min(1.0, connection.weight + strength_increment)
            connection.usage_count += 1
    
    def weaken_all_connections(self, decay_factor: float = 0.01):
        '''Apply decay to all connections - called when new memories are accessed'''
        for connection in self.connections.values():
            connection.weight = max(0.0, connection.weight - decay_factor)
    
    def get_strongest_connections(self, limit: int = 10) -> List[Connection]:
        '''Get connections ordered by weight (strongest first)'''
        return sorted(
            self.connections.values(),
            key=lambda c: c.weight,
            reverse=True
        )[:limit]
    
    def update_access(self):
        '''Update access metadata when node is accessed'''
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        '''Convert to dictionary for storage'''
        return {
            "id": self.id,
            "concept": self.concept,
            "keywords": self.keywords,
            "tags": self.tags,
            "summary": self.summary,
            "full_content": self.full_content,
            "connections": {
                k: {
                    "target_node_id": v.target_node_id,
                    "connection_type": v.connection_type.value,
                    "weight": v.weight,
                    "usage_count": v.usage_count
                }
                for k, v in self.connections.items()
            },
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "embedding_id": self.embedding_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        '''Create MemoryNode from dictionary'''
        # Convert connection data back to Connection objects
        connections = {}
        for k, v in data.get("connections", {}).items():
            connections[k] = Connection(
                target_node_id=v["target_node_id"],
                connection_type=ConnectionType(v["connection_type"]),
                weight=v["weight"],
                usage_count=v["usage_count"]
            )
        
        return cls(
            id=data["id"],
            concept=data["concept"],
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            summary=data["summary"],
            full_content=data["full_content"],
            connections=connections,
            access_count=data.get("access_count", 0),
            importance_score=data.get("importance_score", 0.5),
            embedding_id=data.get("embedding_id")
        )