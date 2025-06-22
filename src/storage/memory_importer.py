"""
Memory Importer - Import existing data into the memory system
Handles importing conversations, documents, and knowledge bases.
"""

import json
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
# from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.memory_node import MemoryNode, Connection, ConnectionType
from ..core.memory_graph import MemoryGraph
from .graph_database import MemoryGraphDatabase
from .vector_store import MemoryVectorStore
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ImportFormat(str, Enum):
    """Supported import formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    CONVERSATION = "conversation"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class ImportResult:
    """Result of an import operation."""
    total_processed: int
    successful_imports: int
    failed_imports: int
    new_connections: int
    errors: List[str]
    imported_memory_ids: List[str]


class MemoryImporter:
    """
    Import various data formats into the memory system.
    """
    
    def __init__(
        self,
        graph_db: Optional[MemoryGraphDatabase] = None,
        vector_store: Optional[MemoryVectorStore] = None,
        memory_graph: Optional[MemoryGraph] = None
    ):
        """Initialize importer with storage backends."""
        self.graph_db = graph_db or MemoryGraphDatabase()
        self.vector_store = vector_store or MemoryVectorStore()
        self.memory_graph = memory_graph or MemoryGraph()
        
        # Import configuration
        self.min_similarity_threshold = settings.import_similarity_threshold if hasattr(settings, 'import_similarity_threshold') else 0.85
        self.auto_connect_threshold = settings.auto_connect_threshold if hasattr(settings, 'auto_connect_threshold') else 0.7
        self.batch_size = 100
    
    def import_file(self, file_path: str, format: ImportFormat) -> ImportResult:
        """
        Import data from a file.
        
        Args:
            file_path: Path to the file to import
            format: Format of the file
            
        Returns:
            ImportResult with statistics
        """
        path = Path(file_path)
        if not path.exists():
            return ImportResult(0, 0, 0, 0, [f"File not found: {file_path}"], [])
        
        logger.info(f"Importing {format.value} file: {file_path}")
        
        if format == ImportFormat.JSON:
            return self._import_json(path)
        elif format == ImportFormat.CSV:
            return self._import_csv(path)
        elif format == ImportFormat.MARKDOWN:
            return self._import_markdown(path)
        elif format == ImportFormat.CONVERSATION:
            return self._import_conversation(path)
        elif format == ImportFormat.KNOWLEDGE_BASE:
            return self._import_knowledge_base(path)
        else:
            return ImportResult(0, 0, 0, 0, [f"Unsupported format: {format}"], [])
    
    def _import_json(self, file_path: Path) -> ImportResult:
        """Import memories from JSON file."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                memories_data = data
            elif isinstance(data, dict) and 'memories' in data:
                memories_data = data['memories']
            else:
                result.errors.append("Invalid JSON structure. Expected list or dict with 'memories' key.")
                return result
            
            # Process memories in batches
            for i in range(0, len(memories_data), self.batch_size):
                batch = memories_data[i:i + self.batch_size]
                batch_result = self._process_memory_batch(batch)
                
                result.total_processed += batch_result.total_processed
                result.successful_imports += batch_result.successful_imports
                result.failed_imports += batch_result.failed_imports
                result.new_connections += batch_result.new_connections
                result.errors.extend(batch_result.errors)
                result.imported_memory_ids.extend(batch_result.imported_memory_ids)
            
        except Exception as e:
            result.errors.append(f"Failed to import JSON: {str(e)}")
            logger.error(f"JSON import failed: {e}")
        
        return result
    
    def _import_csv(self, file_path: Path) -> ImportResult:
        """Import memories from CSV file."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                batch = []
                for row in reader:
                    # Convert CSV row to memory data
                    memory_data = {
                        'concept': row.get('concept', ''),
                        'keywords': [k.strip() for k in row.get('keywords', '').split(',') if k.strip()],
                        'tags': [t.strip() for t in row.get('tags', '').split(',') if t.strip()],
                        'summary': row.get('summary', ''),
                        'full_content': row.get('full_content', row.get('content', '')),
                        'importance_score': float(row.get('importance_score', 0.5))
                    }
                    
                    batch.append(memory_data)
                    
                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        batch_result = self._process_memory_batch(batch)
                        result.total_processed += batch_result.total_processed
                        result.successful_imports += batch_result.successful_imports
                        result.failed_imports += batch_result.failed_imports
                        result.new_connections += batch_result.new_connections
                        result.errors.extend(batch_result.errors)
                        result.imported_memory_ids.extend(batch_result.imported_memory_ids)
                        batch = []
                
                # Process remaining items
                if batch:
                    batch_result = self._process_memory_batch(batch)
                    result.total_processed += batch_result.total_processed
                    result.successful_imports += batch_result.successful_imports
                    result.failed_imports += batch_result.failed_imports
                    result.new_connections += batch_result.new_connections
                    result.errors.extend(batch_result.errors)
                    result.imported_memory_ids.extend(batch_result.imported_memory_ids)
        
        except Exception as e:
            result.errors.append(f"Failed to import CSV: {str(e)}")
            logger.error(f"CSV import failed: {e}")
        
        return result
    
    def _import_markdown(self, file_path: Path) -> ImportResult:
        """Import memories from Markdown file."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse markdown structure
            memories_data = self._parse_markdown_to_memories(content)
            
            # Process all parsed memories
            batch_result = self._process_memory_batch(memories_data)
            result.total_processed = batch_result.total_processed
            result.successful_imports = batch_result.successful_imports
            result.failed_imports = batch_result.failed_imports
            result.new_connections = batch_result.new_connections
            result.errors = batch_result.errors
            result.imported_memory_ids = batch_result.imported_memory_ids
            
        except Exception as e:
            result.errors.append(f"Failed to import Markdown: {str(e)}")
            logger.error(f"Markdown import failed: {e}")
        
        return result
    
    def _import_conversation(self, file_path: Path) -> ImportResult:
        """Import memories from conversation transcript."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract memories from conversation
            memories_data = self._extract_memories_from_conversation(data)
            
            # Process extracted memories
            batch_result = self._process_memory_batch(memories_data)
            result.total_processed = batch_result.total_processed
            result.successful_imports = batch_result.successful_imports
            result.failed_imports = batch_result.failed_imports
            result.new_connections = batch_result.new_connections
            result.errors = batch_result.errors
            result.imported_memory_ids = batch_result.imported_memory_ids
            
        except Exception as e:
            result.errors.append(f"Failed to import conversation: {str(e)}")
            logger.error(f"Conversation import failed: {e}")
        
        return result
    
    def _import_knowledge_base(self, file_path: Path) -> ImportResult:
        """Import structured knowledge base."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            # Process topics/articles
            if 'topics' in kb_data:
                for topic in kb_data['topics']:
                    topic_result = self._process_knowledge_topic(topic)
                    result.total_processed += topic_result.total_processed
                    result.successful_imports += topic_result.successful_imports
                    result.failed_imports += topic_result.failed_imports
                    result.new_connections += topic_result.new_connections
                    result.errors.extend(topic_result.errors)
                    result.imported_memory_ids.extend(topic_result.imported_memory_ids)
            
        except Exception as e:
            result.errors.append(f"Failed to import knowledge base: {str(e)}")
            logger.error(f"Knowledge base import failed: {e}")
        
        return result
    
    def _process_memory_batch(self, memories_data: List[Dict[str, Any]]) -> ImportResult:
        """Process a batch of memory data."""
        result = ImportResult(0, 0, 0, 0, [], [])
        memory_nodes = []
        
        for data in memories_data:
            result.total_processed += 1
            
            try:
                # Check if memory already exists (avoid duplicates)
                existing_memory = self._find_existing_memory(data)
                
                if existing_memory:
                    # Update existing memory if needed
                    updated = self._update_memory_if_needed(existing_memory, data)
                    if updated:
                        result.successful_imports += 1
                        result.imported_memory_ids.append(existing_memory.id)
                    else:
                        result.failed_imports += 1
                else:
                    # Create new memory
                    memory_node = self._create_memory_from_data(data)
                    memory_nodes.append(memory_node)
                    
            except Exception as e:
                result.failed_imports += 1
                result.errors.append(f"Failed to process memory: {str(e)}")
                logger.error(f"Memory processing failed: {e}")
        
        # Batch store new memories
        if memory_nodes:
            store_result = self._batch_store_memories(memory_nodes)
            result.successful_imports += store_result['successful']
            result.failed_imports += store_result['failed']
            result.imported_memory_ids.extend(store_result['stored_ids'])
            
            # Create connections between new memories
            connections = self._create_auto_connections(memory_nodes)
            result.new_connections = len(connections)
        
        return result
    
    def _create_memory_from_data(self, data: Dict[str, Any]) -> MemoryNode:
        """Create a MemoryNode from imported data."""
        return MemoryNode(
            concept=data.get('concept', ''),
            keywords=data.get('keywords', []),
            tags=data.get('tags', []),
            summary=data.get('summary', ''),
            full_content=data.get('full_content', data.get('content', '')),
            importance_score=data.get('importance_score', 0.5),
        )
    
    def _find_existing_memory(self, data: Dict[str, Any]) -> Optional[MemoryNode]:
        """Check if a similar memory already exists."""
        # First check by concept
        concept = data.get('concept', '')
        if concept:
            existing = self.graph_db.find_memories_by_concept(concept)
            if existing:
                # Check similarity
                for memory in existing:
                    if self._calculate_memory_similarity(memory, data) >= self.min_similarity_threshold:
                        return memory
        
        # Check by semantic search
        search_text = f"{concept} {data.get('summary', '')}"
        if search_text.strip():
            results = self.vector_store.semantic_search(
                search_text,
                limit=5,
                min_certainty=self.min_similarity_threshold
            )
            
            for memory_id, certainty, _ in results:
                if certainty >= self.min_similarity_threshold:
                    return self.graph_db.get_memory_node(memory_id)
        
        return None
    
    def _calculate_memory_similarity(self, memory: MemoryNode, data: Dict[str, Any]) -> float:
        """Calculate similarity between existing memory and new data."""
        score = 0.0
        weights = {'concept': 0.4, 'keywords': 0.3, 'summary': 0.3}
        
        # Concept similarity
        if memory.concept.lower() == data.get('concept', '').lower():
            score += weights['concept']
        
        # Keyword overlap
        existing_keywords = set(k.lower() for k in memory.keywords)
        new_keywords = set(k.lower() for k in data.get('keywords', []))
        if existing_keywords and new_keywords:
            overlap = len(existing_keywords & new_keywords) / len(existing_keywords | new_keywords)
            score += weights['keywords'] * overlap
        
        # Summary similarity (simple word overlap)
        existing_words = set(memory.summary.lower().split())
        new_words = set(data.get('summary', '').lower().split())
        if existing_words and new_words:
            overlap = len(existing_words & new_words) / len(existing_words | new_words)
            score += weights['summary'] * overlap
        
        return score
    
    def _update_memory_if_needed(self, memory: MemoryNode, data: Dict[str, Any]) -> bool:
        """Update existing memory with new data if needed."""
        updated = False
        
        # Add new keywords
        existing_keywords = set(memory.keywords)
        new_keywords = set(data.get('keywords', [])) - existing_keywords
        if new_keywords:
            memory.keywords.extend(list(new_keywords))
            updated = True
        
        # Add new tags
        existing_tags = set(memory.tags)
        new_tags = set(data.get('tags', [])) - existing_tags
        if new_tags:
            memory.tags.extend(list(new_tags))
            updated = True
        
        # Update content if longer
        new_content = data.get('full_content', data.get('content', ''))
        if new_content and len(new_content) > len(memory.full_content):
            memory.full_content = new_content
            updated = True
        
        if updated:
            # Update access tracking
            memory.access_count += 1
            
            # Store updates
            self.graph_db.store_memory_node(memory)
            self.vector_store.store_memory_embedding(memory)
        
        return updated
    
    def _batch_store_memories(self, memory_nodes: List[MemoryNode]) -> Dict[str, Any]:
        """Store multiple memories efficiently."""
        stored_ids = []
        successful = 0
        failed = 0
        
        # Store in graph database
        for node in memory_nodes:
            if self.graph_db.store_memory_node(node):
                stored_ids.append(node.id)
                successful += 1
            else:
                failed += 1
        
        # Batch store embeddings
        if stored_ids:
            embedding_results = self.vector_store.batch_store_embeddings(
                [n for n in memory_nodes if n.id in stored_ids]
            )
            
            # Update counts based on embedding results
            for memory_id, success in embedding_results.items():
                if not success:
                    successful -= 1
                    failed += 1
                    stored_ids.remove(memory_id)
        
        return {
            'successful': successful,
            'failed': failed,
            'stored_ids': stored_ids
        }
    
    def _create_auto_connections(self, memory_nodes: List[MemoryNode]) -> List[Tuple[str, str, Connection]]:
        """Automatically create connections between related memories."""
        connections = []
        
        for i, node1 in enumerate(memory_nodes):
            # Find similar memories in the system
            similar = self.vector_store.find_similar_memories(
                node1.id,
                limit=5,
                min_certainty=self.auto_connect_threshold
            )
            
            for memory_id, certainty, _ in similar:
                if certainty >= self.auto_connect_threshold:
                    # Determine connection type based on content
                    connection_type = self._determine_connection_type(node1, memory_id)
                    
                    # Create connection
                    connection = Connection(
                        target_node_id=memory_id,
                        connection_type=connection_type,
                        weight=certainty * 0.5  # Start with moderate weight
                    )
                    
                    self.graph_db.store_connection(node1.id, memory_id, connection)
                    connections.append((node1.id, memory_id, connection))
        
        return connections
    
    def _determine_connection_type(self, node1: MemoryNode, node2_id: str) -> ConnectionType:
        """Determine the type of connection between two memories."""
        # This is a simplified version - could be enhanced with NLP
        node2 = self.graph_db.get_memory_node(node2_id)
        if not node2:
            return ConnectionType.SIMILARITY
        
        # Check for temporal relationships
        if 'before' in node1.concept.lower() or 'after' in node2.concept.lower():
            return ConnectionType.TEMPORAL_BEFORE
        
        # Check for cause/effect
        if any(word in node1.concept.lower() for word in ['cause', 'because', 'leads to']):
            return ConnectionType.CAUSE_EFFECT
        
        # Check for general/specific
        if len(node1.keywords) > len(node2.keywords) * 2:
            return ConnectionType.GENERAL_SPECIFIC
        elif len(node2.keywords) > len(node1.keywords) * 2:
            return ConnectionType.SPECIFIC_GENERAL
        
        # Default to similarity
        return ConnectionType.SIMILARITY
    
    def _parse_markdown_to_memories(self, content: str) -> List[Dict[str, Any]]:
        """Parse markdown content into memory data."""
        memories = []
        lines = content.split('\n')
        
        current_memory = None
        current_section = []
        
        for line in lines:
            if line.startswith('# '):  # Main heading
                if current_memory:
                    current_memory['full_content'] = '\n'.join(current_section)
                    memories.append(current_memory)
                
                current_memory = {
                    'concept': line[2:].strip(),
                    'keywords': [],
                    'tags': [],
                    'summary': '',
                    'importance_score': 0.7
                }
                current_section = []
                
            elif line.startswith('## ') and current_memory:  # Subheading
                current_memory['keywords'].append(line[3:].strip().lower())
                
            elif line.startswith('**Keywords:**') and current_memory:
                keywords = line.replace('**Keywords:**', '').strip()
                current_memory['keywords'].extend([k.strip() for k in keywords.split(',')])
                
            elif line.startswith('**Tags:**') and current_memory:
                tags = line.replace('**Tags:**', '').strip()
                current_memory['tags'].extend([t.strip() for t in tags.split(',')])
                
            elif line.strip() and current_memory:
                current_section.append(line)
                if not current_memory['summary'] and len(line) > 20:
                    current_memory['summary'] = line[:200]
        
        # Add last memory
        if current_memory:
            current_memory['full_content'] = '\n'.join(current_section)
            memories.append(current_memory)
        
        return memories
    
    def _extract_memories_from_conversation(self, conversation_data: Any) -> List[Dict[str, Any]]:
        """Extract memories from conversation data."""
        memories = []
        
        # Handle different conversation formats
        messages = []
        if isinstance(conversation_data, list):
            messages = conversation_data
        elif isinstance(conversation_data, dict) and 'messages' in conversation_data:
            messages = conversation_data['messages']
        
        # Extract key information from messages
        for i, message in enumerate(messages):
            if isinstance(message, dict):
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                
                # Extract concepts from assistant responses
                if role == 'assistant' and len(content) > 50:
                    # Simple extraction - could be enhanced with NLP
                    concepts = self._extract_concepts_from_text(content)
                    
                    for concept in concepts:
                        memory = {
                            'concept': concept['concept'],
                            'keywords': concept['keywords'],
                            'tags': ['conversation', role],
                            'summary': concept['summary'],
                            'full_content': content,
                            'importance_score': 0.6
                        }
                        memories.append(memory)
        
        return memories
    
    def _extract_concepts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract concepts from text (simplified version)."""
        concepts = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs[:3]:  # Limit to first 3 paragraphs
            # Extract first sentence as concept
            sentences = para.split('.')
            if sentences:
                concept = sentences[0].strip()
                
                # Extract keywords (simple word frequency)
                words = para.lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 4:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Sort keywords by frequency
                keywords = sorted(list(word_freq.keys()), key=lambda x: word_freq[x], reverse=True)[:5]
                
                concepts.append({
                    'concept': concept[:100],  # Limit length
                    'keywords': keywords,
                    'summary': para[:200]
                })
        
        return concepts
    
    def _process_knowledge_topic(self, topic: Dict[str, Any]) -> ImportResult:
        """Process a knowledge base topic."""
        result = ImportResult(0, 0, 0, 0, [], [])
        
        # Create parent memory for topic
        topic_memory = {
            'concept': topic.get('title', ''),
            'keywords': topic.get('keywords', []),
            'tags': topic.get('categories', []),
            'summary': topic.get('summary', ''),
            'full_content': topic.get('content', ''),
            'importance_score': topic.get('importance', 0.8)
        }
        
        parent_result = self._process_memory_batch([topic_memory])
        result.total_processed += parent_result.total_processed
        result.successful_imports += parent_result.successful_imports
        result.failed_imports += parent_result.failed_imports
        result.imported_memory_ids.extend(parent_result.imported_memory_ids)
        
        # Process subtopics
        if 'subtopics' in topic and parent_result.imported_memory_ids:
            parent_id = parent_result.imported_memory_ids[0]
            
            for subtopic in topic['subtopics']:
                subtopic_memory = {
                    'concept': subtopic.get('title', ''),
                    'keywords': subtopic.get('keywords', []),
                    'tags': topic.get('categories', []),
                    'summary': subtopic.get('summary', ''),
                    'full_content': subtopic.get('content', ''),
                    'importance_score': subtopic.get('importance', 0.6)
                }
                
                sub_result = self._process_memory_batch([subtopic_memory])
                result.total_processed += sub_result.total_processed
                result.successful_imports += sub_result.successful_imports
                result.failed_imports += sub_result.failed_imports
                
                # Create parent-child connection
                if sub_result.imported_memory_ids:
                    child_id = sub_result.imported_memory_ids[0]
                    connection = Connection(
                        target_node_id=child_id,
                        connection_type=ConnectionType.GENERAL_SPECIFIC,
                        weight=0.8
                    )
                    self.graph_db.store_connection(parent_id, child_id, connection)
                    result.new_connections += 1
                    result.imported_memory_ids.append(child_id)
        
        return result