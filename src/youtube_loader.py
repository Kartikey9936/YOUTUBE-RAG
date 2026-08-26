import re
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character video ID from a YouTube URL.
    Supports standard, short (youtu.be), embed, and mobile URLs.
    """
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL: Could not extract Video ID")

def load_transcript(video_id: str) -> tuple[list[dict], str]:
    """
    Fetches transcript for a given video ID with language fallback (English -> Hindi -> Any available).
    Returns a tuple of (raw_transcript_list, concatenated_text).
    Raises TranscriptsDisabled, NoTranscriptFound, or Exception on failures.
    """
    api = YouTubeTranscriptApi()
    try:
        # Try fetching preferred languages first (English, Hindi)
        fetched = api.fetch(video_id, languages=['en', 'en-US', 'en-GB', 'hi'])
    except NoTranscriptFound:
        # Fallback to any transcript language available for this video
        try:
            fetched = api.fetch(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise e
        except Exception as e:
            raise Exception(f"Could not retrieve transcript: {str(e)}")
    except TranscriptsDisabled as e:
        raise e
    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {str(e)}")

    raw_data = fetched.to_raw_data()
    
    # Convert into standard dictionary items
    transcript_list = []
    for item in raw_data:
        if isinstance(item, dict):
            transcript_list.append(item)
        else:
            transcript_list.append({
                "text": getattr(item, 'text', str(item)),
                "start": getattr(item, 'start', 0.0),
                "duration": getattr(item, 'duration', 0.0)
            })

    full_text = " ".join(chunk["text"].replace("\n", " ") for chunk in transcript_list)
    return transcript_list, full_text


