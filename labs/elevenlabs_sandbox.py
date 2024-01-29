from elevenlabs import generate, play

audio = generate(
  text="""
  Language is a process of free creation; its laws and principles are fixed, 
  but the manner in which the principles of generation are used is free and infinitely varied.
  """,
  voice="Lily",
  model="eleven_multilingual_v1"
)

play(audio)