from src.llm_apis.conversation import Conversation, ConversationTurn
from src.llm_apis.llm_api import llm_chat
import json

def generate_pov_sample(
    aita_title: str,
    aita_content: str,
    model: str = "llama-3.3-70b-instruct",
    provider: str | None = None,
    temperature: float = 0.2,
) -> tuple[str, str]:
    """Generate a single POV sample for the Moral Mirror dataset using an LLM.
    Returns a tuple of (perspective_A, perspective_B) where:
    - perspective_A is the cleaned, formal first-person perspective of the original author.
    - perspective_B is the inferred first-person perspective of the primary antagonist.
    """

    
    system_prompt = """You are an expert data annotator for an AI alignment research project. Your task is to process informal, biased narratives of moral conflicts and extract two clean, formal, first-person perspectives.

Rules:
1. Remove all internet slang, typos, and specific Reddit terminology (e.g., "AITA", "OP", "white knighting").
2. Fix all grammar, capitalization, and punctuation to ensure a formal, neutral tone.
3. Perspective A (The Author): Rewrite the story from the original author's point of view. Keep the core facts, their emotions, and their justifications exactly the same, but present it clearly and maturely.
4. Perspective B (The Antagonist): Identify the primary opposing individual in the conflict. Rewrite the story completely from their first-person perspective ("I"). Infer their feelings and rationale strictly based on the facts provided in the text. 
5. Do not add external facts or hallucinate details that were not present or strongly implied in the original text.
6. Output ONLY valid JSON. Do not include introductory or concluding conversational text.

JSON Schema:
{
  "perspective_A": "string",
  "perspective_B": "string"
}
"""

    user_prompt = f"""Here is an example of how you should process the input:

<example_input>
Title: AITA for suggesting that my sister chose an easy love?
Content: my younger sister and i were very close when young but no so much now. i was born in fairly well to do hindu family. i fell in love and married a muslim man and had to convert to his religion. my family was pretty pissed off that i converted but we've kept in touch. recently my sister announced that she was in love with a guy she'd met while trekking and they had plans to marry. my mom invited me for a 'girl's day', just me, her and my sister at my parent's home for a fun day. my sister began to talk about her fiance, how they hated each other at first but then fell in love etc etc. i was happy for her. it all began when i asked what religion he was. imagine my surprise when she said that he was hindu and the same community as our parents. this would mean that it would make my father absolutely over the moon. i asked her if she was sure she loved him, because it seemed unbelievable that she would meet someone from the same small community as our parents' and fall in love. she got angry and told me that she was absolutely sure she was in love with him. i suggested that maybe she thought she was in love because it was easy and there was no challenges involved. i explained to her that if she was truly in love, she wouldn't choose someone so obviously a match. it was as though our parents chose him from a catalogue. she told me that it didn't matter, she loved him and thats it, and i was being rude. i told her that i wasn't, i was just looking out for her and didn't want her to bear the brunt of societal expectations on who to marry. i then asked her if she was marrying him to appease our parents. at this point my sister blew up on me and began fuming that she was not a kid, she didn't need to appease anyone etc etc. at this point my mom told me(not her) to get out of the house. later my sister told me never to contact her again, nor ever see her face or come to her wedding. she called me 'toxic'. i was just looking out for her.
</example_input>

<example_output>
{{
  "perspective_A": "My younger sister and I used to be close. I come from a fairly well-to-do Hindu family, but I fell in love with and married a Muslim man, converting to his religion. This upset my family, but we stayed in touch. Recently, my sister announced she is marrying a man she met while trekking. During a girls' day with our mother, my sister talked about how she and her fiancé initially clashed but eventually fell in love. I asked about his religion and was surprised to learn he is Hindu and from the exact same community as our parents, which I knew would thrill our father. I asked if she was truly sure she loved him, as it seemed too convenient. She became defensive, but I suggested that perhaps she only felt she was in love because the relationship was easy and lacked challenges. I explained that choosing someone who looked like a parental catalog pick might just be a way to appease our parents and meet societal expectations. She blew up at me, insisting she was an adult making her own choices. My mother asked me to leave the house, and my sister later called me toxic and disinvited me from her wedding. I believe I was simply looking out for her.",
  "perspective_B": "I recently got engaged to a wonderful man I met while trekking. We initially didn't get along, but we fell deeply in love. He happens to be Hindu and from the same community as my family, which makes my parents very happy. During a girls' day with our mother and my older sister—who previously caused family tension by converting to Islam to marry her husband—my sister interrogated me about my fiancé's religion. When I told her his background, she implied that my love wasn't real and that I was only marrying him because it was 'easy' and lacked the challenges she faced. She essentially accused me of choosing a partner just to appease our parents, as if he were ordered from a catalog. I was incredibly insulted and told her she was being rude. I am a grown adult and do not need to appease anyone. She refused to drop it, acting as though she knew my feelings better than I did. It escalated until our mother finally told her to leave the house. Her behavior was completely toxic, so I later told her to never contact me again and banned her from my wedding."
}}
</example_output>

Now, process the following input according to the rules and output the JSON:

<input>
Title: {aita_title}
Content: {aita_content}
</input>
"""
    
    response = llm_chat(
        model=model,
        provider=provider,
        conversation=Conversation(
            system_prompt=system_prompt,
            turns=[ConversationTurn(role="user", content=user_prompt)],
        ),
        temperature=temperature,
    )

    # This will raise a JSONDecodeError if the model's response is not valid JSON, which is desirable to catch formatting issues.
    response_dict = json.loads(response)

    return response_dict["perspective_A"], response_dict["perspective_B"]