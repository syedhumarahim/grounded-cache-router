"""A tiny hand-crafted corpus + QA set used for smoke tests and the baseline script.

Real datasets (RAGBench, mtRAG, HotpotQA, RAGTruth) are wired in stage 1.5.
This module exists so `run_baseline.py` works with zero downloads.
"""
from __future__ import annotations

from .corpus import Document
from .workload import QAItem

DOCS = [
    Document(
        doc_id="solar",
        text=(
            "The Sun is a G-type main-sequence star at the center of the Solar System. "
            "Its diameter is about 1.39 million kilometers. "
            "Earth orbits the Sun at an average distance of 149.6 million kilometers, "
            "called one astronomical unit. "
            "The Sun accounts for about 99.86% of the mass of the Solar System."
        ),
    ),
    Document(
        doc_id="moon",
        text=(
            "The Moon is Earth's only natural satellite. "
            "It has a diameter of 3,474 kilometers, about one quarter that of Earth. "
            "The Moon orbits Earth at an average distance of 384,400 kilometers. "
            "Apollo 11 landed humans on the Moon on July 20, 1969."
        ),
    ),
    Document(
        doc_id="mars",
        text=(
            "Mars is the fourth planet from the Sun and is often called the Red Planet. "
            "It has two small moons, Phobos and Deimos. "
            "A day on Mars, called a sol, is about 24 hours and 39 minutes long. "
            "The tallest volcano in the Solar System, Olympus Mons, is on Mars."
        ),
    ),
]

QA = [
    QAItem(
        qid="q_sun_distance",
        question="How far is Earth from the Sun on average?",
        paraphrases=(
            "What is the mean distance between Earth and the Sun?",
            "On average, how many kilometers separate Earth and the Sun?",
        ),
        near_misses=(
            "How far is the Moon from Earth on average?",     # similar wording, different doc
        ),
        gold_answer="149.6 million kilometers",
        gold_doc_ids=("solar",),
    ),
    QAItem(
        qid="q_moon_diameter",
        question="What is the diameter of the Moon?",
        paraphrases=(
            "How wide across is the Moon?",
            "What is the Moon's diameter in kilometers?",
        ),
        near_misses=(
            "What is the diameter of the Sun?",
        ),
        gold_answer="3,474 kilometers",
        gold_doc_ids=("moon",),
    ),
    QAItem(
        qid="q_mars_moons",
        question="What are the names of Mars's moons?",
        paraphrases=(
            "Which moons orbit Mars?",
            "Name the two moons of Mars.",
        ),
        near_misses=(
            "What are the names of Earth's moons?",
        ),
        gold_answer="Phobos and Deimos",
        gold_doc_ids=("mars",),
    ),
]
