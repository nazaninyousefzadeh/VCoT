"""
VCoT dataset `target` field format (see build_dataset.py).

    "{click_tokens} | <sep> | {answer}"

- Before `` | <sep> | ``: click sequence as ``<click>x,y</click>`` tokens.
- After `` | <sep> | ``: the short answer (e.g. chart QA label).
"""

TARGET_SEP = " | <sep> | "


def parse_vcot_target(target: str) -> tuple[str, str]:
    """
    Split a dataset target string into (clicks_part, answer_part).

    If the separator is missing, returns (target stripped, "").
    """
    if TARGET_SEP not in target:
        return target.strip(), ""
    clicks, answer = target.split(TARGET_SEP, 1)
    return clicks.strip(), answer.strip()
