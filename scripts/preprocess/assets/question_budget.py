"""Fit complete questions to the annotator's token budget using head/tail excerpts."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
import re

from .text import _csv_record_chunks


DEFAULT_QUESTION_TOKENS = 24_576
DEFAULT_QUESTION_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
QUESTION_BUDGET_CATEGORIES = frozenset({'asset-text', 'asset-application'})


@dataclass(frozen=True)
class _ExcerptSource:
    opening: str
    closing: str
    head: str
    tail: str | None
    original_chars: int

    def render(self, head, tail, scope):
        omitted = self.original_chars - len(head) - len(tail)
        marker = (f'\n<{scope}_truncation mode="head_tail" reason="question_token_budget" '
                  f'omitted_chars="{omitted}" />\n')
        return self.opening + head + marker + tail + self.closing


class _PreparedText(str):
    """A bounded string retaining the original omission count for subsequent fitting."""

    def __new__(cls, value, source):
        instance = super().__new__(cls, value)
        instance.source = source
        return instance


class QuestionBudget:
    """Budget after cache lookup: one asset can receive different space in different items."""

    def __init__(self, max_tokens=DEFAULT_QUESTION_TOKENS, tokenizer_name=DEFAULT_QUESTION_TOKENIZER,
                 *, tokenizer=None, local_files_only=False):
        if max_tokens < 256:
            raise ValueError("question token budget must be at least 256")
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        self._tokenizer = tokenizer
        self.local_files_only = local_files_only
        self.tokenizer_char_limit = max_tokens * 8
        self._count_cached = lru_cache(maxsize=128)(self._count)
        self._trim_cached = lru_cache(maxsize=256)(self._fit)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, use_fast=True, local_files_only=self.local_files_only,
            )
        return self._tokenizer

    @property
    def settings(self):
        return {"max_tokens": self.max_tokens, "tokenizer_name": self.tokenizer_name,
                "policy": "head-tail-shared-v2"}

    def count(self, text):
        # Explicit callers can count arbitrary strings, but never retain giant
        # source dumps as LRU keys. The fitting path only counts bounded strings.
        if len(text) > self.tokenizer_char_limit:
            return self._count(text)
        return self._count_cached(text)

    def _count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    @staticmethod
    def _frame(text, scope):
        opening, closing = "", ""
        match = re.match(r"\A(<(?P<tag>[\w-]+)\b[^>]*>\n)", text[:4096])
        if scope == "asset" and match:
            end = "\n</" + match["tag"] + ">"
            if text.endswith(end):
                opening, closing = match[0], end
        return opening, closing, len(opening), len(text) - len(closing)

    def prepare(self, text, scope="asset"):
        """Take bounded source windows before tokenization, copying only the two ends.

        At the default limit, the tokenizer sees at most 196,608 source characters
        per asset, even if a cached SQLite dump contains hundreds of millions.
        The original body length survives fitting, so omission counts stay exact.
        """
        if isinstance(text, _PreparedText) or len(text) <= self.tokenizer_char_limit:
            return text
        opening, closing, start, end = self._frame(text, scope)
        allowance = self.tokenizer_char_limit - len(opening) - len(closing) - 256
        if allowance < 2:
            opening, closing, start, end = "", "", 0, len(text)
            allowance = self.tokenizer_char_limit - 256
        head_chars = (allowance + 1) // 2
        tail_chars = allowance // 2
        head = text[start:start + head_chars]
        tail = text[end - tail_chars:end]
        # These searches only scan the small windows, never the original dump.
        boundary = head.rfind("\n")
        if boundary >= len(head) // 2:
            head = head[:boundary + 1]
        boundary = tail.find("\n")
        if 0 <= boundary < len(tail) // 2:
            tail = tail[boundary + 1:]
        source = _ExcerptSource(opening, closing, head, tail, end - start)
        return _PreparedText(source.render(head, tail, scope), source)

    def trim(self, text, budget, scope="asset"):
        """Fit a bounded excerpt, reusing nearby allowances across repeated assets."""
        text = self.prepare(text, scope)
        if self.count(text) <= budget:
            return text
        # At most 127 tokens of slack buys reuse across questions with slightly
        # different instructions/CSV attachments. Tiny allowances remain exact.
        allowance = budget - budget % 128 if budget >= 1024 else budget
        return self._trim_cached(text, allowance, scope)

    def _fit(self, text, budget, scope):
        if isinstance(text, _PreparedText):
            source = text.source
        else:
            opening, closing, start, end = self._frame(text, scope)
            source = _ExcerptSource(opening, closing, text[start:end], None, end - start)
        head_body = source.head
        tail_body = source.tail if source.tail is not None else head_body

        # Snap to complete lines/CSV records where possible. Huge individual
        # records fall back to Unicode character boundaries, never broken UTF-8.
        def boundaries(body):
            chunks = _csv_record_chunks(body) if source.opening.startswith("<csv_asset ") else body.splitlines(True)
            ends = [0]
            for chunk in chunks:
                ends.append(ends[-1] + len(chunk))
            return ends

        head_boundaries = boundaries(head_body)
        tail_boundaries = boundaries(tail_body) if source.tail is not None else head_boundaries

        def render(keep):
            first = min((keep + 1) // 2, len(head_body))
            last = max(0, len(tail_body) - keep // 2)
            head_end = head_boundaries[bisect_right(head_boundaries, first) - 1]
            tail_start = tail_boundaries[bisect_left(tail_boundaries, last)]
            if head_end and tail_start < len(tail_body):
                first, last = head_end, tail_start
            return source.render(head_body[:first], tail_body[last:], scope)

        best = render(0)
        if self.count(best) > budget:
            raise ValueError(f"Token allowance {budget} cannot hold the {scope} truncation marker")
        available = len(head_body) + len(tail_body) if source.tail is not None else len(head_body) - 1
        low, high = 1, available
        while low <= high:
            keep = (low + high) // 2
            candidate = render(keep)
            if self.count(candidate) <= budget:
                best = candidate
                low = keep + 1
            else:
                high = keep - 1
        return best

    @staticmethod
    def _allocate(demands, available):
        """Share space equally, giving unused allocations from small files to larger ones."""
        allocations = [0] * len(demands)
        ordered = sorted(range(len(demands)), key=demands.__getitem__)
        for position, index in enumerate(ordered):
            remaining = len(ordered) - position
            share = available // remaining
            if demands[index] <= share:
                allocations[index] = demands[index]
                available -= demands[index]
            else:
                extra = available % remaining
                for offset, pending in enumerate(ordered[position:]):
                    allocations[pending] = share + (offset < extra)
                break
        return allocations

    def assemble(self, question, fragments):
        """Enforce the final cap, including base text, all asset types, separators and markers."""
        question = self.prepare(question, "question")
        fragments = [self.prepare(fragment) for fragment in fragments]
        base_pretrimmed = isinstance(question, _PreparedText)
        asset_pretrimmed = [isinstance(fragment, _PreparedText) for fragment in fragments]
        stats = {"question_tokens": 0, "truncated_assets": sum(asset_pretrimmed),
                 "omitted_assets": 0, "question_text_truncated": base_pretrimmed,
                 "was_truncated": base_pretrimmed or any(asset_pretrimmed)}
        combined_chars = len(question) + sum(len(fragment) + 2 for fragment in fragments)
        if combined_chars <= self.tokenizer_char_limit:
            full = question + ("\n\n" + "\n\n".join(fragments) if fragments else "")
            full_count = self.count(full)
            if full_count <= self.max_tokens:
                stats["question_tokens"] = full_count
                return full, stats
        stats["was_truncated"] = True
        if not fragments:
            result = self.trim(question, self.max_tokens, "question")
            stats.update(question_tokens=self.count(result), question_text_truncated=True)
            return result, stats

        demands = [self.count(fragment) for fragment in fragments]
        # Preserve the base question unless assets need space. An oversized base
        # can use half the window; a small collection of assets requires less.
        base_allowance = self.max_tokens - min(sum(demands), self.max_tokens // 2) - 64
        base = self.trim(question, base_allowance, "question")
        stats["question_text_truncated"] = base_pretrimmed or base != question
        available = self.max_tokens - self.count(base) - 64
        separator_cost = max(1, self.count("\n\n"))
        minimums = [min(demand, 96) + separator_cost for demand in demands]

        def selected_indices(count):
            head = (count + 1) // 2
            tail = count // 2
            return list(range(head)) + (list(range(len(fragments) - tail, len(fragments))) if tail else [])

        for _ in range(8):
            indices = list(range(len(fragments)))
            if sum(minimums) > available:
                # Thousands of assets cannot each have a meaningful excerpt in
                # 24K tokens. Retain the first/last files and disclose the gap.
                low, high, count = 0, len(fragments), 0
                while low <= high:
                    middle = (low + high) // 2
                    if sum(minimums[i] for i in selected_indices(middle)) <= available:
                        count = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                indices = selected_indices(count)
            omitted = len(fragments) - len(indices)
            allocations = self._allocate([demands[i] for i in indices],
                                         available - separator_cost * len(indices))
            excerpts = [self.trim(fragments[i], allowance) for i, allowance in zip(indices, allocations)]
            if omitted:
                excerpts.insert((len(indices) + 1) // 2,
                                f'<asset_group_truncation mode="head_tail" omitted_assets="{omitted}" />')
            result = base + "\n\n" + "\n\n".join(excerpts)
            count = self.count(result)
            if count <= self.max_tokens:
                stats.update(question_tokens=count, omitted_assets=omitted,
                             truncated_assets=sum(asset_pretrimmed[i] or demands[i] > allowance
                                                  for i, allowance in zip(indices, allocations)))
                return result, stats
            # Tokenization across boundaries can differ from separate counts.
            # Measure the complete string and reduce the allowance until it fits.
            available -= count - self.max_tokens + 32
        raise ValueError("Could not fit the assembled question within its token budget")
