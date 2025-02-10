from typing import List, Type, Optional
from pydantic import BaseModel, Field


class SchemaRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SchemaRegistry, cls).__new__(cls, *args, **kwargs)
            cls._instance._schemas = {}
        return cls._instance

    def register(self, name: str, schema: Type[BaseModel]):
        self._schemas[name] = schema

    def get(self, name: str) -> Type[BaseModel]:
        return self._schemas.get(name)


class AmbiguityCheck(BaseModel):
    is_ambiguous: bool = Field(..., description="Binary response, indicates if the sentence is ambiguous or not based on world knowledge.")
    #disambiguations: List[str] = Field(description="If the sentence is ambiguous, provide a list of the two disambiguations/interpretations of the sentence.")


schema_registry = SchemaRegistry()
schema_registry.register("ambiguity_check", AmbiguityCheck)

