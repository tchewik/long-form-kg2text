from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base_pipeline import BasePipeline, PipelineConfig, format_triples_as_text


@dataclass
class CoTConfig(PipelineConfig):
    system_template: str = (
        "Act as a system that describes all nodes of the graph connected by edges as a Wikipedia-like text. "
        "Refer only to items explicitly present in the graph, using no external information. "
    )
    user_template: str = (
        "Triples:\n{triples}\n\n"
        "First, reason step by step about what these triples mean.\n"
        "Then, on a new line, write 'Final:' followed by text "
        "concisely and fluently summarizing all the facts."
    )


class CoTPipeline(BasePipeline):
    def __init__(self, lm, dataset, config: Optional[CoTConfig] = None):
        super().__init__(lm=lm, dataset=dataset, config=config or CoTConfig(), dedup_by_graph=True)

    def prompt_fingerprint(self) -> Dict[str, Any]:
        return {
            "pipeline": "cot",
            "system": self.config.system_template,
            "user_template": self.config.user_template,
        }

    def build_prompt(self, triples: List[Dict[str, str]]) -> str:
        triples_text = format_triples_as_text(triples)
        few_shot = self.build_few_shot_prefix()
        user = few_shot + self.config.user_template.format(triples=triples_text)
        return f"{self.config.system_template}\n<|USER_PROMPT|>\n{user}\n\nAnswer: Let's think step-by-step:"

    def extract_final(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]

        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith("final:"):
                # Case 1: "Final: answer"
                after_colon = stripped.split(":", 1)[1].strip()
                if after_colon:
                    return after_colon

                # Case 2: "Final:\n answer"
                remainder = "\n".join(lines[i + 1:]).strip()
                if remainder:
                    return remainder

                return ""
        return text.strip()
