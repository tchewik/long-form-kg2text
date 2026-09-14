from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_pipeline import BasePipeline, PipelineConfig, PromptStyle, format_triples_as_text


@dataclass
class DirectConfig(PipelineConfig):
    prompt_style: PromptStyle = "wikipedia"

    system_template_wikipedia: str = (
        "You are a text generator that produces exactly one concise, fluent, Wikipedia-like paragraph covering all "
        "facts present in the input triples. Do not output any reasoning, analysis, planning, or internal thoughts. "
        "Do not describe your process.  Output only the paragraph."
        "Refer only to items explicitly present in the graph, using no external information."
    )
    user_template_wikipedia: str = (
        "Triples:\n{triples}\n\n"
        "Final:"
    )

    system_template_paper_abstract: str = ()
    user_template_paper_abstract: str = ()


class DirectPipeline(BasePipeline):
    def __init__(self, lm, dataset, config: Optional[DirectConfig] = None):
        super().__init__(lm=lm, dataset=dataset, config=config or DirectConfig(), dedup_by_graph=True)
        if not self.config.stop_strings:
            self.config.stop_strings = ["\n\n"]

    def _get_templates(self) -> Tuple[str, str]:
        style = (self.config.prompt_style or "wikipedia").lower()
        if style == "paper_abstract":
            return (self.config.system_template_paper_abstract, self.config.user_template_paper_abstract)
        return (self.config.system_template_wikipedia, self.config.user_template_wikipedia)

    def prompt_fingerprint(self) -> Dict[str, Any]:
        system_t, user_t = self._get_templates()
        return {
            "pipeline": "direct",
            "prompt_style": self.config.prompt_style,
            "system": system_t,
            "user_template": user_t,
        }

    def build_prompt(self, triples: List[Dict[str, str]]) -> str:
        system_t, user_t = self._get_templates()
        triples_text = format_triples_as_text(triples)
        few_shot = self.build_few_shot_prefix()
        user = few_shot + user_t.format(triples=triples_text)
        return f"{system_t}\n<|USER_PROMPT|>\n{user} "

    def build_messages(self, triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        system_t, user_t = self._get_templates()
        triples_text = format_triples_as_text(triples)
        few_shot = self.build_few_shot_prefix()
        user = few_shot + user_t.format(triples=triples_text)

        return [
            {"role": "system", "content": system_t},
            {"role": "user", "content": user},
        ]

    def extract_final(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]

        for line in text.splitlines():
            if line.strip().lower().startswith("final:"):
                return line.split(":", 1)[1].strip()
        return text.strip()
