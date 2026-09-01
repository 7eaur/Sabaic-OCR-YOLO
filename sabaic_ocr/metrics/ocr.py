from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class EditCounts:
    matches: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            matches=self.matches + other.matches,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein alignment with operation traceback."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                candidates = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(candidates, key=lambda x: x[0])

    counts = EditCounts()
    i, j = n, m
    while i > 0 or j > 0:
        action = op[i][j]
        if action == "M":
            counts.matches += 1
            i -= 1
            j -= 1
        elif action == "S":
            counts.substitutions += 1
            i -= 1
            j -= 1
        elif action == "D":
            counts.deletions += 1
            i -= 1
        elif action == "I":
            counts.insertions += 1
            j -= 1
        else:
            break
    return counts


def normalize_ocr_text(text: str) -> str:
    return "".join(text.split())


def words_from_sabaic(text: str, separator: str = "𐩽") -> List[str]:
    # A physical line break is not necessarily a word boundary in Old South
    # Arabian: attested words can be broken across line boundaries. Therefore
    # only the encoded vertical separator U+10A7D establishes a word boundary.
    compact = "".join(text.split())
    return [w for w in compact.split(separator) if w]


def evaluate_pair(reference: str, hypothesis: str, separator: str = "𐩽") -> dict:
    ref_chars = list(normalize_ocr_text(reference))
    hyp_chars = list(normalize_ocr_text(hypothesis))
    char_counts = edit_counts(ref_chars, hyp_chars)

    ref_words = words_from_sabaic(reference, separator)
    hyp_words = words_from_sabaic(hypothesis, separator)
    word_counts = edit_counts(ref_words, hyp_words)

    cer = char_counts.errors / max(1, len(ref_chars))
    wer = word_counts.errors / max(1, len(ref_words))

    return {
        "character": {
            "reference_count": len(ref_chars),
            "predicted_count": len(hyp_chars),
            "correct": char_counts.matches,
            "wrong": char_counts.errors,
            "substitutions": char_counts.substitutions,
            "deletions": char_counts.deletions,
            "insertions": char_counts.insertions,
            "cer": cer,
            "accuracy_from_cer": max(0.0, 1.0 - cer),
            "match_accuracy": char_counts.matches / max(1, len(ref_chars)),
        },
        "word": {
            "reference_count": len(ref_words),
            "predicted_count": len(hyp_words),
            "correct": word_counts.matches,
            "wrong": word_counts.errors,
            "substitutions": word_counts.substitutions,
            "deletions": word_counts.deletions,
            "insertions": word_counts.insertions,
            "wer": wer,
            "accuracy_from_wer": max(0.0, 1.0 - wer),
            "match_accuracy": word_counts.matches / max(1, len(ref_words)),
        },
    }


def evaluate_corpus(pairs: Iterable[Tuple[str, str]], separator: str = "𐩽") -> dict:
    char_total = EditCounts()
    word_total = EditCounts()
    ref_chars_total = 0
    hyp_chars_total = 0
    ref_words_total = 0
    hyp_words_total = 0
    samples = 0

    for reference, hypothesis in pairs:
        samples += 1
        ref_chars = list(normalize_ocr_text(reference))
        hyp_chars = list(normalize_ocr_text(hypothesis))
        char_total = char_total + edit_counts(ref_chars, hyp_chars)
        ref_chars_total += len(ref_chars)
        hyp_chars_total += len(hyp_chars)

        ref_words = words_from_sabaic(reference, separator)
        hyp_words = words_from_sabaic(hypothesis, separator)
        word_total = word_total + edit_counts(ref_words, hyp_words)
        ref_words_total += len(ref_words)
        hyp_words_total += len(hyp_words)

    cer = char_total.errors / max(1, ref_chars_total)
    wer = word_total.errors / max(1, ref_words_total)
    return {
        "samples": samples,
        "character": {
            "reference_count": ref_chars_total,
            "predicted_count": hyp_chars_total,
            "correct": char_total.matches,
            "wrong": char_total.errors,
            "substitutions": char_total.substitutions,
            "deletions": char_total.deletions,
            "insertions": char_total.insertions,
            "cer": cer,
            "accuracy_from_cer": max(0.0, 1.0 - cer),
            "match_accuracy": char_total.matches / max(1, ref_chars_total),
        },
        "word": {
            "reference_count": ref_words_total,
            "predicted_count": hyp_words_total,
            "correct": word_total.matches,
            "wrong": word_total.errors,
            "substitutions": word_total.substitutions,
            "deletions": word_total.deletions,
            "insertions": word_total.insertions,
            "wer": wer,
            "accuracy_from_wer": max(0.0, 1.0 - wer),
            "match_accuracy": word_total.matches / max(1, ref_words_total),
        },
    }
