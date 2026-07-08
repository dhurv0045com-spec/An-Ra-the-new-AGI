"""Blinded Elo paired-comparison evaluation harness."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable
import time


@dataclass
class EloRating:
    checkpoint_id: str
    rating: float
    matches: int


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate the expected probability of A beating B."""
    return 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, actual_score_a: float, k_factor: float = 32.0) -> tuple[float, float]:
    """Update Elo ratings for a pair after a match.
    
    actual_score_a is 1.0 for A win, 0.5 for draw, 0.0 for B win.
    """
    expected_a = calculate_expected_score(rating_a, rating_b)
    expected_b = calculate_expected_score(rating_b, rating_a)
    
    actual_score_b = 1.0 - actual_score_a
    
    new_rating_a = rating_a + k_factor * (actual_score_a - expected_a)
    new_rating_b = rating_b + k_factor * (actual_score_b - expected_b)
    
    return new_rating_a, new_rating_b


class EloHarness:
    """Manages blinded paired comparisons to establish regression CI baselines."""

    def __init__(self, initial_rating: float = 1200.0) -> None:
        self.ratings: dict[str, EloRating] = {}
        self.initial_rating = initial_rating
        self.history: list[dict[str, object]] = []

    def get_rating(self, checkpoint_id: str) -> EloRating:
        if checkpoint_id not in self.ratings:
            self.ratings[checkpoint_id] = EloRating(checkpoint_id, self.initial_rating, 0)
        return self.ratings[checkpoint_id]

    def _blind_generators(self, gen_a: Callable, gen_b: Callable) -> tuple[Callable, Callable, bool]:
        """Returns two generators, potentially swapped, and a boolean indicating if they were swapped."""
        swapped = random.choice([True, False])
        if swapped:
            return gen_b, gen_a, swapped
        return gen_a, gen_b, swapped

    def run_paired_comparison(
        self,
        checkpoint_a: str,
        generator_a: Callable[[str], str],
        checkpoint_b: str,
        generator_b: Callable[[str], str],
        prompt: str,
        judge: Callable[[str, str, str], float],
    ) -> float:
        """Run a single blinded comparison.
        
        The judge should return 1.0 if the first generated text wins, 0.5 for a draw, and 0.0 if the second wins.
        This function handles the blinding internally.
        """
        blind_gen_1, blind_gen_2, swapped = self._blind_generators(generator_a, generator_b)
        
        response_1 = blind_gen_1(prompt)
        response_2 = blind_gen_2(prompt)
        
        # Judge is unaware of which checkpoint generated which response
        blind_score = judge(prompt, response_1, response_2)
        
        # Unblind the score to attribute it to A
        actual_score_a = 1.0 - blind_score if swapped else blind_score
        
        rating_a = self.get_rating(checkpoint_a)
        rating_b = self.get_rating(checkpoint_b)
        
        new_rating_a, new_rating_b = update_elo(rating_a.rating, rating_b.rating, actual_score_a)
        
        rating_a.rating = new_rating_a
        rating_a.matches += 1
        
        rating_b.rating = new_rating_b
        rating_b.matches += 1
        
        self.history.append({
            "timestamp": time.time(),
            "checkpoint_a": checkpoint_a,
            "checkpoint_b": checkpoint_b,
            "score_a": actual_score_a,
            "new_rating_a": new_rating_a,
            "new_rating_b": new_rating_b,
            "prompt_hash": hash(prompt) # Avoid storing raw prompt for privacy if needed
        })
        
        return actual_score_a

    def check_regression(self, candidate_checkpoint: str, baseline_checkpoint: str, threshold: float = 50.0) -> bool:
        """Evaluate if the candidate regresses against the baseline.
        
        A regression in this context means the candidate has a significantly lower Elo rating
        after sufficient matches (e.g., if we impose a -2% regression block).
        """
        cand_rating = self.get_rating(candidate_checkpoint)
        base_rating = self.get_rating(baseline_checkpoint)
        
        if cand_rating.matches == 0:
            return False # Cannot determine regression without data
            
        # A regression requires sufficient statistical evidence; here we use an Elo drop threshold.
        return cand_rating.rating < base_rating.rating - threshold
