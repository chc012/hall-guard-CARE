# Hallucination Guard with RoBERTa

## Requirements

```bash
pip install transformers[torch]
```

## Usage
```python
from hallguard import HallGuard
clf = HallGuard()
prediction = clf.predict([
    [
        "Hello",
        "Hello",
        "How are you doing?",
        "I'm doing great, how are you?",
        "I like Saturn the most."
    ],
    [
        "I understand. However, there is nothing wrong with telling them that you are sad because you miss them. Something as simple as getting it off your chest can already make you feel better.",
        "I suppose. I just worry that I will make them sad too. It's hard time for everyone.",
        "Totally, but remember than they are there for you whenever you need them.",
        "Yes that's true. Do you have any suggestions on what to say?",
        "I bet something as simple as ""Hey babe, I really miss you. I can't be glad with your abscence"" could make your boyfriend feel loved while at the same time relieving you.",
        "That is true. I believe I have a problem showing vulnerability."
    ]
])

# prediction = [{'label': 0, 'score': 0.0006168214022181928}, {'label': 1, 'score': 0.9376410841941833}]
```

## Details

See `HallGuard` doctrings.
