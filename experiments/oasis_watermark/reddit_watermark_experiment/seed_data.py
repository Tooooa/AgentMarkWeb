# -*- coding: utf-8 -*-
"""
r/TechFuture Seed Data
5 initial posts + 2-3 comments each
"""

SEED_POSTS = [
    {
        "title": "Impact of AI on Programmer Employment",
        "content": """With the rapid adoption of AI coding assistants (like GitHub Copilot, Cursor), the job market for programmers is changing.

On one hand, AI tools significantly boost development efficiency; on the other hand, there are concerns that junior developer roles might be replaced.

What do you think? Is AI a helpful assistant for programmers, or a potential competitor?""",
        "initial_comments": [
            {
                "content": "I think junior roles will indeed be affected, but senior architects and system designers will become even more important. AI can write code, but it can't make decisions.",
                "upvotes": 15
            },
            {
                "content": "AI is just a tool, the key is problem-solving ability. The gap between programmers who use AI vs those who don't will widen.",
                "upvotes": 23
            }
        ]
    },
    {
        "title": "Viability of SpaceX Mars Mission",
        "content": """Starship's latest test progress is impressive, successfully achieving the chopstick catch.

But is Elon Musk's goal of manned Mars landing by 2030 really feasible? Technology, cost, human health risks... challenges seem everywhere.

When do you think humans will truly set foot on Mars?""",
        "initial_comments": [
            {
                "content": "Technically I think it's feasible, but the cost and risks are still huge. NASA's conservative timeline might be more realistic.",
                "upvotes": 12
            },
            {
                "content": "I'm more concerned about the ethical issues of Mars colonization. Who decides who goes? How are resources distributed? These problems are harder than technology.",
                "upvotes": 8
            },
            {
                "content": "SpaceX's iteration speed is truly amazing. 5 years ago no one thought reusable rockets could achieve this.",
                "upvotes": 18
            }
        ]
    },
    {
        "title": "Is Web3 Dead?",
        "content": """The crypto market continues to be sluggish, NFT hype has cooled to freezing point, multiple well-known Web3 projects collapsed...

The Web3 craze of two years ago seems like yesterday, but now it seems no one talks about it.

Is Web3 really dead? Or just finding real application scenarios after the bubble burst?""",
        "initial_comments": [
            {
                "content": "The hype is dead, but the underlying blockchain technology is still developing steadily. DeFi and cross-border payments still have real value.",
                "upvotes": 14
            },
            {
                "content": "Honestly, I think it was a bubble. Now it's time to burst. Truly valuable technology doesn't need this much hype.",
                "upvotes": 21
            }
        ]
    },
    {
        "title": "New VR Device Review: Is Vision Pro Worth Buying?",
        "content": """Apple Vision Pro has been released for almost a year. As an early adopter, I want to share some real user experiences.

Pros: Stunning visuals, natural gesture interaction, excellent ecosystem integration
Cons: Expensive, average wearing comfort, lack of killer apps

Will the VR/AR market change because of Apple's entry? Do you think it's worth buying?""",
        "initial_comments": [
            {
                "content": "The price of 25k is too expensive, ordinary consumers simply can't accept it. Wait for the second generation price drop.",
                "upvotes": 25
            },
            {
                "content": "The experience is indeed stunning, but the app ecosystem is a big problem. Without enough killer apps, even the best hardware is useless.",
                "upvotes": 16
            },
            {
                "content": "I think this is more like a developer preview, not a consumer product. Apple is laying out the future.",
                "upvotes": 11
            }
        ]
    },
    {
        "title": "Open Source vs Closed Source Models: Who is the Future of AI?",
        "content": """Open source models like Llama 3, Mistral, Qwen are progressing rapidly, closing the gap with GPT-4, Claude.

The open source camp advocates transparency, controllability, community driven;
The closed source camp emphasizes safety, performance, commercial sustainability.

Which side are you more optimistic about? Should enterprises choose open source or closed source models?""",
        "initial_comments": [
            {
                "content": "Open source models are indeed progressing fast, but there is still a gap in top-tier reasoning capabilities. However, for most application scenarios, open source is enough.",
                "upvotes": 19
            },
            {
                "content": "For enterprises, data security and controllability are more important. Local deployment of open source models is the trend.",
                "upvotes": 22
            }
        ]
    }
]


def get_seed_post_actions():
    """
    Generate ManualAction list for seed posts
    Returns action list required for environment initialization
    """
    actions = []
    for post in SEED_POSTS:
        actions.append({
            "type": "CREATE_POST",
            "content": f"[{post['title']}]\n\n{post['content']}"
        })
    return actions


def get_initial_comment_actions():
    """
    Generate ManualAction list for initial comments
    post_id starts from 1
    """
    actions = []
    for post_id, post in enumerate(SEED_POSTS, start=1):
        for comment in post["initial_comments"]:
            actions.append({
                "type": "CREATE_COMMENT",
                "post_id": post_id,
                "content": comment["content"]
            })
    return actions
