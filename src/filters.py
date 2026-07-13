'''
This function is used to filter the chunks that are retrieved from the database

Pipeline in detail:

User query -> Query Embedding -> Query embedded and filters into Qdrant and execute retrieval --> cosine sim ->  return chunks

In Qdrant, each chunk is stored as a point including 3 primary fields: ID, Vector, Payload(Metadata)
After return chunks -> put all retrieved chunks into {context} in prompt -> feed into LLM -> answer 
'''

# pyrefly: ignore [missing-import]
from pydantic import BaseModel, model_validator
# pyrefly: ignore [missing-import]
from qdrant_client.http import models as qmodels
from config import settings

class MetadataFilter(BaseModel):
    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "MetadataFilter":
        
        names = [n.strip() for n in (self.filenames or []) if isinstance(n, str) and n.strip()]

        #Handle conflict in filename and filenames
        if not names:
            self.filenames = None
        elif len(names) == 1:
            self.filename, self.filenames = names[0], None
        else:
            self.filename, self.filenames, self.page = None, names, None

        # Remove white space for other fields
        if self.filename is not None:
            self.filename = self.filename.strip() or None
        if self.section is not None:
            self.section = self.section.strip() or None
        if self.document_id is not None:
            self.document_id = self.document_id.strip() or None

        return self


def _coerce_filter(filters):
    """
    Convert filtering input into MetadataFilter object to avoid system errors
    """
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        return filters
    if isinstance(filters, dict):
        return MetadataFilter(**filters)
    return None


def filters_to_dict(filters):
    """
    Convert MetadataFilter object into dictionary and remove empty fields
    """
    f = _coerce_filter(filters)
    return None if f is None else f.model_dump(exclude_none=True) or None


def filters_to_qdrant(filters):
    """
    Translate filtering condition into FieldCondition language of Qdrant
    """
    flat = filters_to_dict(filters)
    if not flat:
        return None
    
    conditions = []
    for field, value in flat.items():
        if field == "filenames" and isinstance(value, list):
            # IN operator: match with any in the list of filenames
            conditions.append(qmodels.FieldCondition(
                key="metadata.filename", match=qmodels.MatchAny(any=value)
            ))
        elif isinstance(value, (str, int)):
            # = operator: exact match with the value
            conditions.append(qmodels.FieldCondition(
                key=f"metadata.{field}", match=qmodels.MatchValue(value=value)
            ))
    
    return qmodels.Filter(must=conditions) if conditions else None