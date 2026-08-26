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
    try:
        # Try fetching using list_transcripts if available, or direct get_transcript
        try:
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try English first
            try:
                transcript_obj = transcript_list_obj.find_transcript(['en', 'en-US', 'en-GB'])
                raw_data = transcript_obj.fetch()
            except NoTranscriptFound:
                # Try Hindi next
                try:
                    transcript_obj = transcript_list_obj.find_transcript(['hi'])
                    raw_data = transcript_obj.fetch()
                except NoTranscriptFound:
                    # Fallback to any generated/manual transcript
                    try:
                        transcript_obj = transcript_list_obj.find_generated_transcript(['en', 'hi'])
                        raw_data = transcript_obj.fetch()
                    except NoTranscriptFound:
                        # Fetch the first available transcript
                        for t in transcript_list_obj:
                            raw_data = t.fetch()
                            break
                        else:
                            raise NoTranscriptFound(video_id, ['en', 'hi'], transcript_list_obj)
        except (AttributeError, Exception) as list_err:
            if isinstance(list_err, (TranscriptsDisabled, NoTranscriptFound)):
                raise list_err
            # Fallback to simple get_transcript call
            try:
                raw_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                raise e
            except Exception:
                raw_data = YouTubeTranscriptApi.get_transcript(video_id)

        # Convert Raw Script objects / dictionaries into standard dict list
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

    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)
    except NoTranscriptFound:
        raise NoTranscriptFound(video_id, ['en', 'hi'], None)
    except Exception as e:
        raise Exception(f"Could not retrieve transcript: {str(e)}")

