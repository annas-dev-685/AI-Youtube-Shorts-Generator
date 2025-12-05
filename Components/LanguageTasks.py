from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API")
# Provider: 'openai' or 'gemini'. Default to 'openai' for backward compatibility.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

# Note: Do not require an API key at import time. We'll validate keys depending on
# the selected provider when creating the LLM instance so the module can be used
# even if only Gemini (or OpenAI) is configured.

class JSONResponse(BaseModel):
    """
    The response should strictly follow the following structure: -
     [
        {
        start: "Start time of the clip",
        content: "Highlight Text",
        end: "End Time for the highlighted clip"
        }
     ]
    """
    start: float = Field(description="Start time of the clip")
    content: str= Field(description="Highlight Text")
    end: float = Field(description="End time for the highlighted clip")

system = """
The input contains a timestamped transcription of a video.
Select a 2-minute segment from the transcription that contains something interesting, useful, surprising, controversial, or thought-provoking.
The selected text should contain only complete sentences.
Do not cut the sentences in the middle.
The selected text should form a complete thought.
Return a JSON object with the following structure:
## Output 
[{{
    start: "Start time of the segment in seconds (number)",
    content: "The transcribed text from the selected segment (clean text only, NO timestamps)",
    end: "End time of the segment in seconds (number)"
}}]

## Input
{Transcription}
"""

# User = """
# Example
# """




def GetHighlight(Transcription):
    # Create LLM based on provider selection (OpenAI or Google Gemini)
    llm = None
    try:
        if LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI
            if not api_key:
                raise ValueError("OPENAI_API not set for OpenAI provider. Set OPENAI_API in .env")
            llm = ChatOpenAI(
                model="gpt-5-nano",
                temperature=1.0,
                api_key=api_key,
            )
        elif LLM_PROVIDER in ("gemini", "google", "google_gemini"):
            # Try a few possible langchain Google Gemini chat model class locations
            gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API")
            if not gemini_api_key:
                raise ValueError("GOOGLE_API_KEY (or GOOGLE_API) not set for Gemini provider. Set it in .env")
            created = False
            try:
                # from langchain.chat_models import ChatGoogleGemini
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key)
                created = True
            except Exception as e:
                print("Could not import ChatGoogleGenerativeAI from langchain_google_genai:", str(e))
                pass
            if not created:
                raise ImportError(
                    "Gemini provider requested but no compatible langchain Google chat model is available. \n"
                    "Install or upgrade the langchain Google/vertex packages or set LLM_PROVIDER=openai."
                )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Supported: openai, gemini")

        print(f"Using LLM provider: {LLM_PROVIDER}")

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("user", Transcription)
            ]
        )

        # Apply structured output using function calling
        structured_llm = llm.with_structured_output(JSONResponse, method="function_calling")
        chain = prompt | structured_llm
        
        print("Calling LLM for highlight selection...")
        response = chain.invoke({"Transcription":Transcription})
        
        # Validate response
        if not response:
            print("ERROR: LLM returned empty response")
            return None, None
        
        if not hasattr(response, 'start') or not hasattr(response, 'end'):
            print(f"ERROR: Invalid response structure: {response}")
            return None, None
        
        try:
            Start = int(response.start)
            End = int(response.end)
        except (ValueError, TypeError) as e:
            print(f"ERROR: Could not parse start/end times from response")
            print(f"  response.start: {response.start}")
            print(f"  response.end: {response.end}")
            print(f"  Error: {e}")
            return None, None
        
        # Validate times
        if Start < 0 or End < 0:
            print(f"ERROR: Negative time values - Start: {Start}s, End: {End}s")
            return None, None
        
        if End <= Start:
            print(f"ERROR: Invalid time range - Start: {Start}s, End: {End}s (end must be > start)")
            return None, None
        
        # Log the selected segment
        print(f"\n{'='*60}")
        print(f"SELECTED SEGMENT DETAILS:")
        print(f"Time: {Start}s - {End}s ({End-Start}s duration)")
        print(f"Content: {response.content}")
        print(f"{'='*60}\n")
        
        if Start==End:
            Ask = input("Error - Get Highlights again (y/n) -> ").lower()
            if Ask == "y":
                Start, End = GetHighlight(Transcription)
            return Start, End
        return Start,End
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN GetHighlight FUNCTION:")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"\nTranscription length: {len(Transcription)} characters")
        print(f"First 200 chars: {Transcription[:200]}...")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print(GetHighlight(User))
