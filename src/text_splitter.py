from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src import config

def split_documents(transcript_list: list[dict], video_id: str, chunk_size: int = None, chunk_overlap: int = None) -> list[Document]:
    """
    Splits the full transcript into chunks using RecursiveCharacterTextSplitter,
    and maps each chunk back to the transcript timestamps to preserve metadata.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    # 1. Build the full transcript text and keep track of character offsets
    full_text_parts = []
    offsets = []
    current_char = 0

    for item in transcript_list:
        text = item["text"].replace("\n", " ").strip()
        if not text:
            continue
        length = len(text)
        start_time = item["start"]
        end_time = start_time + item.get("duration", 0.0)
        
        full_text_parts.append(text)
        offsets.append({
            "start_char": current_char,
            "end_char": current_char + length,
            "start_time": start_time,
            "end_time": end_time
        })
        current_char += length + 1 # +1 for the space joining them

    full_transcript_text = " ".join(full_text_parts)

    # 2. Run standard recursive character text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    raw_splits = splitter.split_text(full_transcript_text)

    # 3. Align raw splits back to timestamps
    documents = []
    search_start_idx = 0

    for i, split_text in enumerate(raw_splits):
        # Find index in full_transcript_text to calculate character range
        # We start searching from search_start_idx to ensure correct order
        start_idx = full_transcript_text.find(split_text, search_start_idx)
        if start_idx == -1:
            # Fallback to searching from start if not found sequentially
            start_idx = full_transcript_text.find(split_text)
        
        if start_idx != -1:
            end_idx = start_idx + len(split_text)
            # Update sequential search pointer
            # Slide it forward slightly, allowing overlap matching
            search_start_idx = max(0, start_idx + 1)
        else:
            start_idx = 0
            end_idx = len(split_text)

        # Find overlapping transcript timestamps
        overlapping_start_times = []
        overlapping_end_times = []

        for offset in offsets:
            # Check if this transcript entry overlaps with the split character range
            # Range overlap condition: max(start1, start2) < min(end1, end2)
            if max(start_idx, offset["start_char"]) < min(end_idx, offset["end_char"]):
                overlapping_start_times.append(offset["start_time"])
                overlapping_end_times.append(offset["end_time"])

        # Determine start/end times
        start_time = overlapping_start_times[0] if overlapping_start_times else 0.0
        end_time = overlapping_end_times[-1] if overlapping_end_times else 0.0

        # Construct metadata
        metadata = {
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "chunk_id": i,
            "start_time": start_time,
            "end_time": end_time
        }

        documents.append(Document(page_content=split_text, metadata=metadata))

    return documents
