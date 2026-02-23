from __future__ import annotations

import asyncio
from typing import List

from empathy_engine.api.schemas import SynthesisRequest
from empathy_engine.pipeline.engine import get_pipeline


async def run_demo_cases() -> None:
    pipeline = get_pipeline()

    cases: List[SynthesisRequest] = [
        # SynthesisRequest(
        #     text="I just got the job I always wanted! This is fantastic!!!",
        #     session_id="demo-joy",
        #     format="mp3",
        #     return_debug=True,
        # ),
        SynthesisRequest(
            text="I feel really down today. Nothing seems to be working out.",
            session_id="demo-sad",
            format="mp3",
            return_debug=True,
        ),
        # SynthesisRequest(
        #     text="Are you serious? You lost the files again?!",
        #     session_id="demo-anger",
        #     format="mp3",
        #     return_debug=True,
        # ),
        # SynthesisRequest(
        #     text=(
        #         "At first I was worried about the presentation, "
        #         "but as I started speaking, I grew more confident and even excited."
        #     ),
        #     session_id="demo-arc",
        #     format="mp3",
        #     return_debug=True,
        # ),
        
        # SynthesisRequest(
        #     text="This is FANTASTIC!!!",
        #     session_id="demo-intensity",
        #     format="mp3",
        #     return_debug=True,
        # ),
    ]

    
    print("Running primary demo cases...")
    for req in cases:
        resp = await pipeline.synthesize(req)
        print(f"Text: {req.text}")
        print(f" -> Saved to: {resp.file_path}")
        if resp.debug:
            print(f"    Primary emotion: {resp.debug.primary_emotion}")
            print(f"    Intensity: {resp.debug.intensity:.2f}")
        print()


def main() -> None:
    asyncio.run(run_demo_cases())


if __name__ == "__main__":
    main()

