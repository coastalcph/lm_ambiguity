import csv
import json
import logging
import re
from pathlib import Path, PurePath
from typing import Any, Dict, Optional, Type, Union

from jinja2 import Environment, FileSystemLoader, Template
from omegaconf import DictConfig, OmegaConf
from outlines.fsm.json_schema import build_regex_from_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TemplateLoader:
    def __init__(
        self,
        template_dir: PurePath = Path(__file__).parent / "templates",
        extension: str = ".jinja2",
    ):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.extension = extension

    def load(self, template_name: str) -> Template:
        return self.env.get_template(f"{template_name}{self.extension}")


class TsvWriter:
    def __init__(self, path: Path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def __enter__(self):
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True)

        self.file = open(self.path, "w")
        self.writer = csv.writer(self.file, delimiter="\t", **self.kwargs)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


def setup_logging() -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s]  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )


def setup_config(config: DictConfig, print: bool = True) -> None:
    OmegaConf.resolve(config)

    if print:
        logger.info(f"\n{50*'-'}\n{OmegaConf.to_yaml(config)}{50*'-'}")


def parse_json(text_response: str) -> Optional[Dict[str, Any]]:
    pattern = r"\{[^{}]*\}"
    matches = list(re.finditer(pattern, text_response))

    if not matches:
        return None

    match = matches[0]
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Extend the search for nested structures
        extended_json_str = _extend_search(text_response, match.span())
        try:
            return json.loads(extended_json_str)
        except json.JSONDecodeError:
            # If all else fails, try to extract the JSON data manually
            data = {}
            pairs = re.findall(r'(".*?")\s*:\s*(.*)', text_response)

            if not pairs:
                return None

            for pair in pairs:
                key = pair[0].strip('"')
                value = pair[1].strip('"')
                data[key] = value
            return data


def _extend_search(text, span):
    # Extend the search to try to capture nested structures
    start, end = span
    nest_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            nest_count += 1
        elif text[i] == "}":
            nest_count -= 1
            if nest_count == 0:
                return text[start : i + 1]
    return text[start:end]


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def validate_json_with_schema(json_data: Dict[str, Any], schema: Type[BaseModel]) -> bool:
    regex = build_regex_from_schema(convert_json_schema_to_str(schema), whitespace_pattern=r" ?")
    return re.fullmatch(regex, json.dumps(json_data)) is not None


def parse_pydantic_schema(pydantic_model: Type[BaseModel]) -> str:
    simple_schema = {}
    raw_schema = pydantic_model.model_json_schema()

    for name, value in raw_schema["properties"].items():
        # For boolean types, we want to display "true/false" instead of a description
        if "type" in value and value["type"] == "boolean":
            simple_schema[name] = "true/false"
        elif "description" in value:
            simple_schema[name] = value["description"]
        else:
            simple_schema[name] = f"<{name}>"

    return json.dumps(simple_schema, indent=2).replace('"true/false"', "true/false")
