"""litm2026 -- a 2026 replication of *Lost in the Middle* (Liu et al., TACL 2024).

Original paper: https://arxiv.org/abs/2307.03172
Original code and data: https://github.com/nelson-liu/lost-in-the-middle (MIT)

Module map:

* :mod:`litm2026.data`      -- loaders for the paper's released data, and the fixed
                               question subset that makes every cell paired
* :mod:`litm2026.prompting` -- the paper's prompt construction, ported, plus the
                               documented chat adaptation
* :mod:`litm2026.scoring`   -- best-subspan EM, ported verbatim
* :mod:`litm2026.providers` -- Anthropic + any OpenAI-compatible endpoint
* :mod:`litm2026.costs`     -- token accounting and the hard spend cap
* :mod:`litm2026.runner`    -- async, resumable, cost-capped experiment runner
* :mod:`litm2026.stats`     -- bootstrap CIs, paired tests, power, the U-shape index
* :mod:`litm2026.plotting`  -- figures (and a refusal to draw a curve without CIs)

STATUS: no valid run yet. Run 1 is kept in ``results/raw/`` but is invalid
(answers truncated at 100 tokens); see "What went wrong in run 1" in README.md.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
