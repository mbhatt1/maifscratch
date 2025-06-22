# Multi-modal Data

MAIF excels at handling multiple data modalities simultaneously - text, images, audio, video, and structured data - with unified semantic understanding and cross-modal relationships.

## Overview

Multi-modal AI systems can process and understand different types of data together, creating richer, more contextual intelligence. MAIF provides native support for:

- **Text & Images**: Document understanding, image captioning, visual question answering
- **Audio & Text**: Speech recognition, audio transcription, voice assistants
- **Video & Audio**: Video analysis, subtitle generation, content moderation
- **Structured Data**: Integration with databases, APIs, and knowledge graphs

```mermaid
graph TB
    subgraph "Multi-modal Processing Pipeline"
        Input[Multi-modal Input]
        
        Input --> Text[Text Processing]
        Input --> Image[Image Processing]
        Input --> Audio[Audio Processing]
        Input --> Video[Video Processing]
        Input --> Structured[Structured Data]
        
        Text --> TextEmbed[Text Embeddings]
        Image --> ImageEmbed[Image Embeddings]
        Audio --> AudioEmbed[Audio Embeddings]
        Video --> VideoEmbed[Video Embeddings]
        Structured --> StructEmbed[Structured Embeddings]
        
        TextEmbed --> Fusion[Cross-modal Fusion]
        ImageEmbed --> Fusion
        AudioEmbed --> Fusion
        VideoEmbed --> Fusion
        StructEmbed --> Fusion
        
        Fusion --> Understanding[Unified Understanding]
        Understanding --> Output[Multi-modal Output]
    end
    
    style Input fill:#3c82f6,stroke:#1e40af,stroke-width:2px,color:#fff
    style Fusion fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style Understanding fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
```

## Core Concepts

### 1. Cross-Modal Embeddings

MAIF automatically generates embeddings that capture relationships between different modalities. The following example demonstrates how adding text, image, and audio content to an artifact enables cross-modal similarity search.

```python
from maif_sdk import create_artifact, create_client
import numpy as np

client = create_client()
# Mock data for demonstration.
sunset_image_data = np.random.rand(100, 100, 3)
ocean_sounds_data = np.random.rand(44100)

# Create a multi-modal artifact.
artifact = create_artifact("multimodal-demo", client)

# Add content of different data types.
text_id = artifact.add_text("A beautiful sunset over the ocean")
image_id = artifact.add_image(sunset_image_data)
audio_id = artifact.add_audio(ocean_sounds_data)

# MAIF automatically identifies and creates relationships between related content.
relationships = artifact.get_relationships(text_id)
print(f"Text block relates to {len(relationships)} other blocks.")

# Search for content similar to the text, but across all modalities.
similar_blocks = artifact.search_similar(text_id, cross_modal=True)
for block in similar_blocks:
    print(f"Similar {block.type}: {block.title} (similarity: {block.similarity:.3f})")
```

### 2. Semantic Alignment

MAIF uses advanced algorithms to align semantic meaning across modalities. This allows you to query for concepts and have MAIF return relevant results regardless of the data type.

```python
# Mock image data.
red_car_image = np.random.rand(100, 100, 3)

# Add text and an image that are semantically related.
artifact.add_text("Red sports car", metadata={"category": "vehicle"})
artifact.add_image(red_car_image, metadata={"category": "vehicle"})

# Query for a general concept and retrieve results from both text and image modalities.
results = artifact.search("fast vehicle", modalities=["text", "image"])
for result in results:
    print(f"{result.type}: {result.title} - Score: {result.relevance:.3f}")
```

### 3. Contextual Understanding

MAIF maintains context across different data types within an artifact, enabling a holistic understanding of conversational or sequential multi-modal data.

```python
# Mock image data.
user_uploaded_image = np.random.rand(100, 100, 3)

# Create an artifact to hold a conversation.
conversation_artifact = create_artifact("conversation", client)

# Add a sequence of text and image blocks representing a conversation.
conversation_artifact.add_text("User: Can you describe this image?")
conversation_artifact.add_image(user_uploaded_image)
conversation_artifact.add_text("AI: This image shows a golden retriever playing in a park")
conversation_artifact.add_text("User: What breed characteristics can you identify?")

# The system can retrieve the full context, including all modalities.
context = conversation_artifact.get_context()
print(f"Conversation has {len(context.blocks)} blocks across {len(context.modalities)} modalities")
```

## Supported Modalities

### Text Processing

MAIF provides advanced natural language understanding features that can be enabled when adding text data.

```python
# Enable specific NLP features when adding a text block.
text_block = artifact.add_text(
    "John Smith from Acme Corp called about the Q4 financial report",
    features={
        "extract_entities": True,   # Named Entity Recognition (NER)
        "detect_pii": True,         # Personally Identifiable Information detection
        "analyze_sentiment": True,  # Sentiment analysis
        "extract_topics": True      # Topic modeling
    }
)

# Access the features extracted by the processing pipeline.
features = text_block.get_features()
print(f"Entities: {features.entities}")
print(f"PII detected: {features.pii_detected}")
print(f"Sentiment: {features.sentiment}")
```

### Image Processing

MAIF offers a range of computer vision capabilities that can be applied to image data.

```python
# Mock image data.
image_data = np.random.rand(100, 100, 3)

# Enable specific computer vision features for an image block.
image_block = artifact.add_image(
    image_data,
    features={
        "detect_objects": True,     # Object detection
        "extract_text": True,       # Optical Character Recognition (OCR)
        "analyze_scene": True,      # Scene understanding
        "detect_faces": False       # Disable face detection for privacy
    }
)

# Access the visual features extracted from the image.
features = image_block.get_features()
print(f"Objects detected: {features.objects}")
print(f"Text in image: {features.extracted_text}")
print(f"Scene description: {features.scene_description}")
```

### Audio Processing

MAIF can process audio data to extract speech, identify speakers, and understand emotional tone.

```python
# Mock audio data.
audio_data = np.random.rand(44100)

# Enable audio analysis features when adding an audio block.
audio_block = artifact.add_audio(
    audio_data,
    sample_rate=44100,
    features={
        "transcribe": True,         # Speech-to-text
        "identify_speaker": True,   # Speaker diarization/identification
        "detect_emotion": True,     # Emotion detection from speech
        "classify_audio": True      # Classify sounds (e.g., music, speech)
    }
)

# Access the features extracted from the audio.
features = audio_block.get_features()
print(f"Transcription: {features.transcription}")
print(f"Speaker: {features.speaker_id}")
print(f"Emotion: {features.emotion}")
```

### Video Processing

MAIF supports comprehensive video analysis, including scene detection, action recognition, and content moderation.

```python
# Mock video data.
video_data = np.random.rand(10, 100, 100, 3) # 10 frames

# Enable video processing features.
video_block = artifact.add_video(
    video_data,
    features={
        "detect_scenes": True,      # Detect scene changes
        "recognize_actions": True,  # Recognize actions within the video
        "track_objects": True,      # Track objects across frames
        "generate_captions": True,  # Generate subtitles/captions
        "moderate_content": True    # Check for sensitive content
    }
)

# Access the features extracted from the video.
features = video_block.get_features()
print(f"Scenes: {len(features.scenes)}")
print(f"Actions: {features.actions}")
print(f"Captions: {features.captions}")
```

## Cross-Modal Use Cases

### 1. Document Understanding

Process documents that contain a mix of text, images, and structured data to achieve a holistic understanding.

```python
# Mock data for a financial report.
report_text = "The Q4 financial performance was strong..."
chart_image = np.random.rand(100, 100, 3)
financial_data = {"revenue": 1000, "profit": 200}

# Create an artifact for the document.
document_artifact = create_artifact("financial-report", client)

# Add the different components of the document to the artifact.
document_artifact.add_text(report_text)
document_artifact.add_image(chart_image, metadata={"type": "chart"})
document_artifact.add_structured_data(financial_data, metadata={"type": "table"})

# Generate a summary that considers all modalities to provide a complete overview.
summary = document_artifact.summarize(
    include_modalities=["text", "image", "structured"],
    focus="financial_performance"
)
print(f"Document summary: {summary}")
```

### 2. Content Moderation

MAIF can perform multi-modal content safety analysis to detect inappropriate content across text, images, and video simultaneously. This is crucial for maintaining safe online platforms.

```python
from maif_sdk import ModerationPolicy

# Define a content moderation policy.
moderation_policy = ModerationPolicy(
    rules={
        "hate_speech": {"threshold": 0.8, "action": "block"},
        "violence": {"threshold": 0.9, "action": "alert"},
        "adult_content": {"threshold": 0.7, "action": "blur"}
    }
)

# Create an artifact with the moderation policy attached.
moderation_artifact = create_artifact(
    "content-feed", client, moderation_policy=moderation_policy
)

# Add multi-modal content to the artifact.
# MAIF will automatically check it against the policy.
moderation_result = moderation_artifact.add_post(
    text="Some user-generated text.",
    image=user_uploaded_image,
    video=user_uploaded_video # Assuming this variable exists
)

# Check the moderation results.
if moderation_result.is_flagged:
    print(f"Content flagged for: {moderation_result.reasons}")
    print(f"Action taken: {moderation_result.action}")
```

### 3. Interactive AI Assistant

Multi-modal AI interactions:

```python
# Interactive assistant
assistant_artifact = create_artifact("ai-assistant", client)

# User sends image with question
assistant_artifact.add_image(user_image)
assistant_artifact.add_text("What's happening in this image?")

# AI processes both modalities
response = assistant_artifact.generate_response(
    context_modalities=["image", "text"],
    response_type="descriptive"
)

assistant_artifact.add_text(response, metadata={"source": "ai"})
```

## Advanced Features

### 1. Cross-Modal Search

Search across different data types:

```python
# Cross-modal search
results = artifact.search(
    query="happy children playing",
    modalities=["text", "image", "video"],
    fusion_method="late_fusion",
    weights={"text": 0.4, "image": 0.4, "video": 0.2}
)

for result in results:
    print(f"{result.modality}: {result.title} (score: {result.score:.3f})")
```

### 2. Multi-Modal Embeddings

Generate unified embeddings across modalities:

```python
# Multi-modal embeddings
embedding = artifact.get_multimodal_embedding(
    block_ids=[text_id, image_id, audio_id],
    fusion_method="attention_weighted"
)

# Use for similarity search
similar_artifacts = client.search_similar_artifacts(
    embedding=embedding,
    modalities=["text", "image", "audio"]
)
```

### 3. Temporal Alignment

Align time-based modalities:

```python
# Temporal alignment for video + audio
video_artifact = create_artifact("presentation", client)

# Add synchronized content
video_artifact.add_video(presentation_video)
video_artifact.add_audio(presentation_audio)
video_artifact.add_text(presentation_transcript)

# Align timestamps
alignment = video_artifact.align_temporal_modalities(
    reference_modality="video",
    target_modalities=["audio", "text"]
)

# Query by time
content_at_time = video_artifact.get_content_at_time(
    timestamp="00:05:30",
    modalities=["video", "audio", "text"]
)
```

## Performance Optimization

### 1. Efficient Processing

Optimize multi-modal processing:

```python
# Batch processing for efficiency
batch_artifact = create_artifact("batch-processing", client)

# Process multiple modalities in parallel
batch_artifact.add_batch([
    {"type": "text", "data": text_data},
    {"type": "image", "data": image_data},
    {"type": "audio", "data": audio_data}
], parallel=True)

# Streaming processing for large files
stream_artifact = create_artifact("streaming", client)
stream_artifact.add_video_stream(
    video_stream,
    chunk_size="10s",
    process_realtime=True
)
```

### 2. Memory Management

Handle large multi-modal datasets:

```python
# Memory-efficient processing
large_artifact = create_artifact("large-dataset", client, 
    config={
        "memory_mapping": True,
        "lazy_loading": True,
        "compression": "high"
    }
)

# Process large video files
large_artifact.add_video(
    large_video_file,
    processing_mode="streaming",
    memory_limit="2GB"
)
```

## Best Practices

### 1. Data Quality

Ensure high-quality multi-modal data:

```python
# Data quality checks
quality_report = artifact.check_data_quality(
    checks={
        "text": ["language_consistency", "encoding_validity"],
        "image": ["resolution_check", "format_validation"],
        "audio": ["sample_rate_consistency", "noise_level"]
    }
)

if not quality_report.passed:
    print(f"Quality issues found: {quality_report.issues}")
```

### 2. Privacy Considerations

Handle sensitive multi-modal data:

```python
# Privacy-aware processing
private_artifact = create_artifact("sensitive-data", client,
    privacy_config={
        "anonymize_faces": True,
        "redact_pii": True,
        "encrypt_audio": True
    }
)

# Add data with privacy protection
private_artifact.add_image(image_with_faces, privacy_level="high")
private_artifact.add_text(text_with_pii, privacy_level="high")
```

### 3. Scalability

Scale multi-modal processing:

```python
# Distributed processing
distributed_artifact = create_artifact("distributed", client,
    config={
        "processing_nodes": 4,
        "load_balancing": "round_robin",
        "fault_tolerance": True
    }
)

# Process large datasets across nodes
distributed_artifact.process_dataset(
    dataset_path="/path/to/multimodal/dataset",
    batch_size=100,
    parallel_workers=8
)
```

## Integration Examples

### 1. With Computer Vision Libraries

```python
import cv2
import torch
from transformers import CLIPModel, CLIPProcessor

# Integrate with CLIP for vision-language understanding
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process image-text pairs
def process_image_text_pair(image, text):
    artifact = create_artifact("clip-processing", client)
    
    # Add raw data
    image_id = artifact.add_image(image)
    text_id = artifact.add_text(text)
    
    # Generate CLIP embeddings
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt")
    outputs = clip_model(**inputs)
    
    # Store embeddings
    image_embedding = outputs.image_embeds.detach().numpy()
    text_embedding = outputs.text_embeds.detach().numpy()
    
    artifact.add_embedding(image_embedding, metadata={"source": "clip_image"})
    artifact.add_embedding(text_embedding, metadata={"source": "clip_text"})
    
    return artifact
```

### 2. With Speech Processing

```python
import whisper
import librosa

# Integrate with Whisper for speech recognition
whisper_model = whisper.load_model("base")

def process_audio_with_whisper(audio_file):
    artifact = create_artifact("speech-processing", client)
    
    # Load audio
    audio_data, sr = librosa.load(audio_file)
    
    # Add raw audio
    audio_id = artifact.add_audio(audio_data, sample_rate=sr)
    
    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_file)
    
    # Add transcription
    text_id = artifact.add_text(result["text"], metadata={
        "source": "whisper_transcription",
        "language": result["language"],
        "confidence": result.get("confidence", 0.0)
    })
    
    # Link audio and text
    artifact.add_relationship(audio_id, text_id, "transcription")
    
    return artifact
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Files**
   ```python
   # Use streaming processing
   artifact.add_video_stream(large_video, chunk_size="30s")
   ```

2. **Slow Cross-Modal Search**
   ```python
   # Pre-compute embeddings
   artifact.precompute_embeddings(modalities=["text", "image"])
   ```

3. **Inconsistent Quality Across Modalities**
   ```python
   # Standardize quality
   artifact.standardize_quality({
       "image": {"min_resolution": (224, 224)},
       "audio": {"sample_rate": 16000},
       "text": {"min_length": 10}
   })
   ```

## Next Steps

- Explore [Semantic Understanding](semantic.md) for advanced AI algorithms
- Learn about [Real-time Processing](streaming.md) for live multi-modal data
- Check out [Performance Optimization](performance.md) for scaling tips
- See [Examples](../examples/) for complete multi-modal applications 