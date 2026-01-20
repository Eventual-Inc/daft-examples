import daft
from daft.functions import prompt, download
from typing import Literal
from pydantic import BaseModel, field_validator, Field

# Schema validation with guardrails
class ProductInfo(BaseModel):
    category: Literal["furniture", "electronics"]
    price_usd: float 
    has_contact_info: bool

class SafeListing(BaseModel):
    clean_text: str

    @field_validator("clean_text")
    @classmethod
    def redact_email(cls, v: str) -> str:
        return re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 
            "[REDACTED]", 
            v
        )

# Sample product listings with PII and images
data = [{
    "post_id": "1234567890",
    "listing": 
        "Vintage wooden chair $50. Email john@email.com",
    "image_url": 
        "https://images.unsplash.com/photo-15064397736"
}]

df = (
    daft.from_pylist(data)
    # Extract structured fields with schema validation
    .with_column(
        "product_info",
        prompt(
            daft.col("listing"),
            return_format=ProductInfo,
            model="gpt-5-mini",
        ).unnest(),
    )
    # Redact PII
    .with_column(
        "safe_listing",
        prompt(
            daft.col("listing"),
            system_message="Remove emails and phone numbers, replacing with [REDACTED].",
            model="gpt-5-mini",
        ),
    )
    # Handle multimodal data: download and decode images
    .with_column(
        "image", 
        download(daft.col("image_url")).decode_image()
    )
)

df.write_parquet("s3://my-bucket/enriched-products")
