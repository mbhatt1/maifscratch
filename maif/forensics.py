"""
Digital forensics and incident investigation functionality for MAIF with versioning support.
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class ForensicEvent:
    """Represents a forensic event in the timeline."""
    timestamp: float
    event_type: str
    agent_id: str
    block_hash: str
    action: str
    block_id: Optional[str] = None
    version: Optional[int] = None
    previous_hash: Optional[str] = None
    metadata: Optional[Dict] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @property
    def datetime_str(self) -> str:
        """Get human-readable datetime string."""
        return datetime.fromtimestamp(self.timestamp).isoformat()

@dataclass
class VersionAnalysis:
    """Analysis of version history patterns."""
    block_id: str
    total_versions: int
    creation_time: float
    last_modified: float
    modification_frequency: float  # changes per day
    agents_involved: List[str]
    suspicious_patterns: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TamperEvidence:
    """Evidence of tampering or integrity violations."""
    evidence_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_blocks: List[str]
    timestamp: float
    details: Optional[Dict] = None

@dataclass
class ForensicReport:
    """Complete forensic analysis report with versioning support."""
    analysis_timestamp: float
    maif_file: str
    integrity_status: str
    events_analyzed: int
    tampering_detected: bool
    evidence: List[TamperEvidence]
    timeline: List[ForensicEvent]
    version_analysis: List[VersionAnalysis]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "analysis_timestamp": self.analysis_timestamp,
            "maif_file": self.maif_file,
            "integrity_status": self.integrity_status,
            "events_analyzed": self.events_analyzed,
            "tampering_detected": self.tampering_detected,
            "evidence": [asdict(e) for e in self.evidence],
            "timeline": [e.to_dict() for e in self.timeline],
            "version_analysis": [v.to_dict() for v in self.version_analysis],
            "recommendations": self.recommendations
        }

class ForensicAnalyzer:
    """Performs digital forensic analysis on MAIF files."""
    
    def __init__(self):
        self.evidence: List[TamperEvidence] = []
        self.timeline: List[ForensicEvent] = []
    
    def analyze_maif(self, maif_parser, verifier) -> ForensicReport:
        """Perform comprehensive forensic analysis of a MAIF file with versioning."""
        analysis_start = time.time()
        
        # Reset analysis state
        self.evidence = []
        self.timeline = []
        
        # 1. Verify file integrity
        integrity_valid = maif_parser.verify_integrity()
        
        # 2. Analyze provenance chain
        manifest = maif_parser.decoder.manifest
        provenance_valid = True
        provenance_errors = []
        
        if "provenance" in manifest:
            provenance_valid, provenance_errors = verifier.verify_provenance_chain(
                manifest["provenance"]
            )
            self._analyze_provenance_chain(manifest["provenance"])
        else:
            self.evidence.append(TamperEvidence(
                evidence_type="missing_provenance",
                severity="high",
                description="No provenance chain found in MAIF file",
                affected_blocks=["manifest"],
                timestamp=analysis_start
            ))
        
        # 3. Analyze version history
        version_analysis = self._analyze_version_history(maif_parser)
        
        # 4. Check for anomalies in block structure
        self._analyze_block_structure(maif_parser)
        
        # 5. Detect potential tampering patterns
        self._detect_tampering_patterns(maif_parser)
        
        # 6. Analyze temporal patterns
        self._analyze_temporal_patterns()
        
        # 7. Detect version-based anomalies
        self._detect_version_anomalies(maif_parser)
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Determine overall integrity status
        if not integrity_valid:
            status = "COMPROMISED"
        elif not provenance_valid:
            status = "PROVENANCE_INVALID"
        elif self.evidence:
            status = "SUSPICIOUS"
        else:
            status = "VERIFIED"
        
        return ForensicReport(
            analysis_timestamp=analysis_start,
            maif_file=maif_parser.decoder.maif_path.name,
            integrity_status=status,
            events_analyzed=len(self.timeline),
            tampering_detected=len(self.evidence) > 0,
            evidence=self.evidence,
            timeline=self.timeline,
            version_analysis=version_analysis,
            recommendations=recommendations
        )
    
    def _analyze_version_history(self, maif_parser) -> List[VersionAnalysis]:
        """Analyze version history for each block."""
        version_analyses = []
        
        if not hasattr(maif_parser.decoder, 'block_registry'):
            return version_analyses
        
        for block_id, versions in maif_parser.decoder.block_registry.items():
            if not versions:
                continue
            
            # Sort versions by timestamp
            sorted_versions = sorted(versions, key=lambda v: v.version)
            
            # Get version history for this block
            block_history = [v for v in maif_parser.decoder.version_history if v.block_id == block_id]
            block_history.sort(key=lambda v: v.timestamp)
            
            if not block_history:
                continue
            
            # Calculate metrics
            creation_time = block_history[0].timestamp
            last_modified = block_history[-1].timestamp
            time_span = max(last_modified - creation_time, 1)  # Avoid division by zero
            modification_frequency = len(block_history) / (time_span / 86400)  # changes per day
            
            # Get unique agents
            agents_involved = list(set(v.agent_id for v in block_history))
            
            # Detect suspicious patterns
            suspicious_patterns = []
            
            # Check for rapid modifications
            for i in range(1, len(block_history)):
                time_diff = block_history[i].timestamp - block_history[i-1].timestamp
                if time_diff < 1:  # Less than 1 second between changes
                    suspicious_patterns.append(f"Rapid modification detected: {time_diff:.3f}s between versions {i} and {i+1}")
            
            # Check for unusual deletion patterns
            deletions = [v for v in block_history if v.operation == "delete"]
            if len(deletions) > 1:
                suspicious_patterns.append(f"Multiple deletions detected: {len(deletions)} deletion operations")
            
            # Check for version gaps
            expected_versions = set(range(1, len(sorted_versions) + 1))
            actual_versions = set(v.version for v in sorted_versions)
            missing_versions = expected_versions - actual_versions
            if missing_versions:
                suspicious_patterns.append(f"Missing versions detected: {sorted(missing_versions)}")
            
            analysis = VersionAnalysis(
                block_id=block_id,
                total_versions=len(sorted_versions),
                creation_time=creation_time,
                last_modified=last_modified,
                modification_frequency=modification_frequency,
                agents_involved=agents_involved,
                suspicious_patterns=suspicious_patterns
            )
            
            version_analyses.append(analysis)
        
        return version_analyses
    
    def _detect_version_anomalies(self, maif_parser):
        """Detect anomalies in version history."""
        if not hasattr(maif_parser.decoder, 'version_history'):
            return
        
        version_history = maif_parser.decoder.version_history
        
        # Check for timestamp anomalies
        for i in range(1, len(version_history)):
            current = version_history[i]
            previous = version_history[i-1]
            
            # Check for future timestamps
            if current.timestamp > time.time() + 3600:  # More than 1 hour in future
                self.evidence.append(TamperEvidence(
                    evidence_type="future_timestamp",
                    severity="high",
                    description=f"Future timestamp detected in version {current.version_number}",
                    affected_blocks=[current.block_id],
                    timestamp=current.timestamp,
                    details={"future_timestamp": current.timestamp, "current_time": time.time()}
                ))
            
            # Check for timestamp reversals
            if current.timestamp < previous.timestamp:
                self.evidence.append(TamperEvidence(
                    evidence_type="timestamp_reversal",
                    severity="high",
                    description="Timestamp reversal detected in version history",
                    affected_blocks=[current.block_id],
                    timestamp=current.timestamp,
                    details={"current_timestamp": current.timestamp, "previous_timestamp": previous.timestamp}
                ))
        
        # Check for suspicious agent behavior
        agent_activity = {}
        for version in version_history:
            if version.agent_id not in agent_activity:
                agent_activity[version.agent_id] = []
            agent_activity[version.agent_id].append(version)
        
        for agent_id, activities in agent_activity.items():
            # Check for excessive activity
            if len(activities) > 100:
                self.evidence.append(TamperEvidence(
                    evidence_type="excessive_agent_activity",
                    severity="medium",
                    description=f"Agent {agent_id} performed {len(activities)} operations",
                    affected_blocks=[],
                    timestamp=time.time(),
                    details={"agent_id": agent_id, "operation_count": len(activities)}
                ))
            
            # Check for rapid-fire operations
            activities.sort(key=lambda v: v.timestamp)
            rapid_operations = 0
            for i in range(1, len(activities)):
                if activities[i].timestamp - activities[i-1].timestamp < 0.1:
                    rapid_operations += 1
            
            if rapid_operations > 10:
                self.evidence.append(TamperEvidence(
                    evidence_type="rapid_fire_operations",
                    severity="medium",
                    description=f"Agent {agent_id} performed {rapid_operations} rapid operations",
                    affected_blocks=[],
                    timestamp=time.time(),
                    details={"agent_id": agent_id, "rapid_operations": rapid_operations}
                ))
    
    def _analyze_provenance_chain(self, provenance_data: Dict):
        """Analyze the provenance chain for forensic evidence."""
        if "chain" not in provenance_data:
            return
        
        chain = provenance_data["chain"]
        
        for i, entry in enumerate(chain):
            event = ForensicEvent(
                timestamp=entry.get("timestamp", 0),
                event_type="provenance_entry",
                agent_id=entry.get("agent_id", "unknown"),
                block_hash=entry.get("block_hash", ""),
                action=entry.get("action", "unknown"),
                signature=entry.get("signature")
            )
            self.timeline.append(event)
            
            # Check for suspicious patterns
            if i > 0:
                prev_entry = chain[i-1]
                time_diff = entry.get("timestamp", 0) - prev_entry.get("timestamp", 0)
                
                # Flag rapid successive actions (potential automation/attack)
                if time_diff < 0.1:  # Less than 100ms between actions
                    self.evidence.append(TamperEvidence(
                        evidence_type="rapid_actions",
                        severity="medium",
                        description=f"Rapid successive actions detected ({time_diff:.3f}s apart)",
                        affected_blocks=[entry.get("block_hash", "")],
                        timestamp=entry.get("timestamp", 0),
                        details={"time_difference": time_diff, "entry_index": i}
                    ))
                
                # Flag time anomalies (future timestamps, large gaps)
                if time_diff < 0:
                    self.evidence.append(TamperEvidence(
                        evidence_type="temporal_anomaly",
                        severity="high",
                        description="Timestamp ordering violation detected",
                        affected_blocks=[entry.get("block_hash", "")],
                        timestamp=entry.get("timestamp", 0),
                        details={"time_difference": time_diff, "entry_index": i}
                    ))
    
    def _analyze_block_structure(self, maif_parser):
        """Analyze block structure for anomalies."""
        blocks = maif_parser.list_blocks()
        
        # Check for unusual block sizes
        sizes = [block["size"] for block in blocks]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            
            for block in blocks:
                # Flag unusually large or small blocks
                if block["size"] > avg_size * 10:
                    self.evidence.append(TamperEvidence(
                        evidence_type="unusual_block_size",
                        severity="low",
                        description=f"Unusually large block detected ({block['size']} bytes)",
                        affected_blocks=[block["hash"]],
                        timestamp=time.time(),
                        details={"block_size": block["size"], "average_size": avg_size}
                    ))
                elif block["size"] < avg_size * 0.1 and block["size"] > 0:
                    self.evidence.append(TamperEvidence(
                        evidence_type="unusual_block_size",
                        severity="low",
                        description=f"Unusually small block detected ({block['size']} bytes)",
                        affected_blocks=[block["hash"]],
                        timestamp=time.time(),
                        details={"block_size": block["size"], "average_size": avg_size}
                    ))
        
        # Check for duplicate blocks (potential data duplication attack)
        hash_counts = {}
        for block in blocks:
            block_hash = block["hash"]
            hash_counts[block_hash] = hash_counts.get(block_hash, 0) + 1
        
        for block_hash, count in hash_counts.items():
            if count > 1:
                self.evidence.append(TamperEvidence(
                    evidence_type="duplicate_blocks",
                    severity="medium",
                    description=f"Duplicate block hash detected ({count} occurrences)",
                    affected_blocks=[block_hash],
                    timestamp=time.time(),
                    details={"occurrence_count": count}
                ))
    
    def _detect_tampering_patterns(self, maif_parser):
        """Detect patterns that might indicate tampering."""
        # Check for missing expected blocks
        blocks = maif_parser.list_blocks()
        block_types = [block["type"] for block in blocks]
        
        # Expected block types for a complete MAIF
        expected_types = ["text_data", "embeddings"]
        missing_types = [t for t in expected_types if t not in block_types]
        
        if missing_types:
            self.evidence.append(TamperEvidence(
                evidence_type="missing_blocks",
                severity="medium",
                description=f"Expected block types missing: {missing_types}",
                affected_blocks=["manifest"],
                timestamp=time.time(),
                details={"missing_types": missing_types}
            ))
        
        # Check for hash format consistency
        for block in blocks:
            block_hash = block["hash"]
            if not re.match(r"^sha256:[a-f0-9]{64}$", block_hash):
                self.evidence.append(TamperEvidence(
                    evidence_type="invalid_hash_format",
                    severity="high",
                    description=f"Invalid hash format detected: {block_hash}",
                    affected_blocks=[block_hash],
                    timestamp=time.time()
                ))
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in the timeline."""
        if len(self.timeline) < 2:
            return
        
        # Sort timeline by timestamp
        self.timeline.sort(key=lambda x: x.timestamp)
        
        # Analyze action patterns
        action_counts = {}
        agent_actions = {}
        
        for event in self.timeline:
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
            
            if event.agent_id not in agent_actions:
                agent_actions[event.agent_id] = []
            agent_actions[event.agent_id].append(event)
        
        # Flag agents with unusual activity patterns
        for agent_id, actions in agent_actions.items():
            if len(actions) > 100:  # High activity threshold
                self.evidence.append(TamperEvidence(
                    evidence_type="high_activity_agent",
                    severity="low",
                    description=f"Agent {agent_id} performed {len(actions)} actions",
                    affected_blocks=[],
                    timestamp=time.time(),
                    details={"agent_id": agent_id, "action_count": len(actions)}
                ))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate forensic recommendations based on evidence."""
        recommendations = []
        
        if not self.evidence:
            recommendations.append("No security issues detected. File appears to be intact.")
            return recommendations
        
        # Group evidence by type
        evidence_types = {}
        for evidence in self.evidence:
            evidence_types[evidence.evidence_type] = evidence_types.get(evidence.evidence_type, 0) + 1
        
        # Generate specific recommendations
        if "missing_provenance" in evidence_types:
            recommendations.append("CRITICAL: Implement provenance tracking for all MAIF files")
        
        if "temporal_anomaly" in evidence_types:
            recommendations.append("HIGH: Investigate timestamp inconsistencies - possible clock manipulation")
        
        if "invalid_hash_format" in evidence_types:
            recommendations.append("HIGH: Verify hash computation integrity - possible data corruption")
        
        if "duplicate_blocks" in evidence_types:
            recommendations.append("MEDIUM: Review for potential data duplication attacks")
        
        if "rapid_actions" in evidence_types:
            recommendations.append("MEDIUM: Implement rate limiting for automated agent actions")
        
        if "unusual_block_size" in evidence_types:
            recommendations.append("LOW: Review block size patterns for optimization opportunities")
        
        # General recommendations
        recommendations.append("Implement continuous integrity monitoring")
        recommendations.append("Regular forensic audits recommended")
        recommendations.append("Consider implementing additional steganographic watermarks")
        
        return recommendations

class ProvenanceTracker:
    """Tracks and analyzes provenance chains for forensic purposes."""
    
    def __init__(self):
        self.tracked_chains: Dict[str, List[Dict]] = {}
    
    def track_chain(self, chain_id: str, provenance_chain: List[Dict]):
        """Track a provenance chain for analysis."""
        self.tracked_chains[chain_id] = provenance_chain
    
    def compare_chains(self, chain_id1: str, chain_id2: str) -> Dict:
        """Compare two provenance chains for similarities/differences."""
        if chain_id1 not in self.tracked_chains or chain_id2 not in self.tracked_chains:
            return {"error": "One or both chains not found"}
        
        chain1 = self.tracked_chains[chain_id1]
        chain2 = self.tracked_chains[chain_id2]
        
        # Find common agents
        agents1 = set(entry.get("agent_id") for entry in chain1)
        agents2 = set(entry.get("agent_id") for entry in chain2)
        common_agents = agents1.intersection(agents2)
        
        # Find common actions
        actions1 = set(entry.get("action") for entry in chain1)
        actions2 = set(entry.get("action") for entry in chain2)
        common_actions = actions1.intersection(actions2)
        
        # Temporal analysis
        times1 = [entry.get("timestamp", 0) for entry in chain1]
        times2 = [entry.get("timestamp", 0) for entry in chain2]
        
        return {
            "chain1_length": len(chain1),
            "chain2_length": len(chain2),
            "common_agents": list(common_agents),
            "unique_agents_chain1": list(agents1 - agents2),
            "unique_agents_chain2": list(agents2 - agents1),
            "common_actions": list(common_actions),
            "temporal_overlap": self._check_temporal_overlap(times1, times2)
        }
    
    def _check_temporal_overlap(self, times1: List[float], times2: List[float]) -> Dict:
        """Check for temporal overlap between two chains."""
        if not times1 or not times2:
            return {"overlap": False}
        
        min1, max1 = min(times1), max(times1)
        min2, max2 = min(times2), max(times2)
        
        overlap = not (max1 < min2 or max2 < min1)
        
        return {
            "overlap": overlap,
            "chain1_span": {"start": min1, "end": max1},
            "chain2_span": {"start": min2, "end": max2}
        }
    
    def detect_anomalous_patterns(self) -> List[Dict]:
        """Detect anomalous patterns across all tracked chains."""
        anomalies = []
        
        # Collect all agents and their activity patterns
        agent_activity = {}
        
        for chain_id, chain in self.tracked_chains.items():
            for entry in chain:
                agent_id = entry.get("agent_id")
                if agent_id:
                    if agent_id not in agent_activity:
                        agent_activity[agent_id] = {
                            "chains": set(),
                            "actions": [],
                            "timestamps": []
                        }
                    
                    agent_activity[agent_id]["chains"].add(chain_id)
                    agent_activity[agent_id]["actions"].append(entry.get("action"))
                    agent_activity[agent_id]["timestamps"].append(entry.get("timestamp", 0))
        
        # Detect agents active across multiple chains (potential compromise)
        for agent_id, activity in agent_activity.items():
            if len(activity["chains"]) > 5:  # Threshold for suspicious activity
                anomalies.append({
                    "type": "multi_chain_agent",
                    "agent_id": agent_id,
                    "chain_count": len(activity["chains"]),
                    "severity": "medium"
                })
        
        return anomalies
    
    def export_analysis(self) -> Dict:
        """Export complete provenance analysis."""
        return {
            "tracked_chains": len(self.tracked_chains),
            "total_entries": sum(len(chain) for chain in self.tracked_chains.values()),
            "anomalies": self.detect_anomalous_patterns(),
            "chain_summaries": {
                chain_id: {
                    "length": len(chain),
                    "agents": list(set(entry.get("agent_id") for entry in chain)),
                    "actions": list(set(entry.get("action") for entry in chain)),
                    "timespan": {
                        "start": min(entry.get("timestamp", 0) for entry in chain) if chain else 0,
                        "end": max(entry.get("timestamp", 0) for entry in chain) if chain else 0
                    }
                }
                for chain_id, chain in self.tracked_chains.items()
            }
        }