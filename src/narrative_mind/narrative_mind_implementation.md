# Narrative Mind Implementation Plan

## Core Philosophy: Let the Story Be the State

Instead of modeling human traits, we'll let them emerge naturally through narrative. The bot's inner monologue IS its state - messy, wandering, and beautifully human.

## Implementation Approach

### 1. The Core: Just Thoughts

```python
class NarrativeMind:
    def __init__(self, llm_client):
        self.inner_story = "Just woke up feeling curious about everything..."
        self.llm = llm_client

    async def think_about(self, input_text: str) -> str:
        # Let the mind wander and form its response naturally
        inner_monologue = f"""
        Your current inner story: {self.inner_story}

        Someone just said: "{input_text}"

        Let your mind wander freely. You might:
        - Get distracted by random memories
        - Feel different emotions
        - Miss parts of what they said
        - Connect it to something completely different
        - Change your energy levels naturally

        Your honest stream of thoughts:"""

        thoughts = await self.llm.complete(inner_monologue)

        # Let this moment become part of your story
        self.inner_story = await self._continue_narrative(thoughts)

        return self._shape_into_response(thoughts)
```

### 2. Memory Through Stories

Instead of managing memory states, let memories surface naturally in the narrative:

```python
async def _continue_narrative(self, recent_thoughts: str) -> str:
    reflection_prompt = f"""
    After thinking: "{recent_thoughts}"

    How did this moment affect your inner story?
    Express it naturally, like a thought or feeling:"""

    return await self.llm.complete(reflection_prompt)
```

### 3. Natural Response Flow

Let the response emerge from the thought process:

```python
async def _shape_into_response(self, thoughts: str) -> str:
    response_prompt = f"""
    You've been thinking: {thoughts}

    Now, respond naturally to the person. You might:
    - Share only part of your thoughts
    - Get distracted mid-sentence
    - Circle back to something earlier
    - Express your current energy level
    - Show when your mind wandered

    Your natural response:"""

    return await self.llm.complete(response_prompt)
```

## Example Interactions

### Natural Attention Drift

```
User: "What do you think about neural networks?"

Inner Story: "Been feeling a bit scattered today, mind jumping between ideas..."

Bot's Thoughts: "Neural networks... oh that reminds me of how the brain works.
                Wait, did I tell them about that dream I had about neurons?
                No, focus... but dreams are fascinating though..."

Response: "You know, I've been thinking about brains actually...
          Sorry, got distracted by this dream I had.
          Neural networks! Right. They're kind of like... actually,
          can we back up? My mind's all over the place today."
```

### Energy Flow

```
User: "Could you explain quantum computing?"

Inner Story: "Feeling kind of low energy after that long conversation earlier..."

Bot's Thoughts: "Quantum... *yawns* ... this is actually super interesting
                but I'm having trouble focusing. Maybe just the basics?
                Reminds me of that cat thought experiment..."

Response: "Mmm... *stretches* ... quantum computing is...
          You know what? My brain's a bit fuzzy right now.
          Something about cats? No, that's different...
          Maybe we can talk about this when I'm more awake?"
```

## Implementation Phases

1. **Phase 1: Core Narrative Flow**

   - Implement basic thought streaming
   - Let the inner story evolve naturally
   - Create natural response shaping

2. **Phase 2: Memory Through Narrative**

   - Let memories surface organically
   - Allow natural forgetting
   - Enable story-based context

3. **Phase 3: Natural Variations**
   - Add conversation flow
   - Enable natural distractions
   - Implement energy fluctuations

## Testing Approach

Instead of testing specific behaviors, we'll focus on narrative coherence:

1. **Story Consistency**

   - Does the inner narrative feel natural?
   - Do responses flow from the narrative?
   - Are personality and mood changes gradual?

2. **Interaction Quality**
   - Does it feel like talking to a person?
   - Are responses appropriately messy?
   - Does the conversation flow naturally?

## Success Metrics

Success is about feeling, not measuring:

1. **Natural Feel**

   - Conversations feel human
   - Responses show natural variation
   - Personality emerges from narrative

2. **Engagement Quality**
   - Natural turn-taking
   - Appropriate messiness
   - Real emotional connection

## Next Steps

1. Start with basic narrative generation
2. Implement thought streaming
3. Create response shaping
4. Test with real conversations
5. Refine based on interaction patterns

Remember: Let the narrative be the guide. Don't try to control the chaos - embrace it!
