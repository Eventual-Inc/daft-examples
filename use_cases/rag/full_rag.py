# /// script
# description = "Full RAG example"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "openai", "pymupdf", "python-dotenv", "numpy"]
# ///

import pymupdf  # type: ignore
import daft
from daft import DataType, col, lit
from daft.functions import embed_text, cosine_distance, file, prompt, unnest
from dotenv import load_dotenv  # type: ignore


@daft.func(
    return_dtype=DataType.list(
        DataType.struct(
            {
                "page_number": DataType.int32(),
                "page_text": DataType.string(),
            }
        )
    )
)
def extract_pdf_pages(pdf_file: daft.File):
    """Extract text from each PDF page as Daft rows."""
    pages = []
    with pdf_file.to_tempfile() as temp_file:
        document = None
        try:
            document = pymupdf.Document(filename=str(temp_file.name), filetype="pdf")
            for page_number, page in enumerate(document):
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(
                        {
                            "page_number": page_number,
                            "page_text": text,
                            "page_image": 
                            "page_audio":
                            "page_video": 
                            "page_raw"
                            "analytics":
                            "metrics": {

                            }
                        }
                    )
        except Exception as exc:
            print(f"Failed to extract PDF contents: {exc}")
        finally:
            if document is not None:
                document.close()
    return pages



@daft.cls()
class SpaCyChunkText:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    @daft.method(
        return_dtype=DataType.list(
            DataType.struct(
                {
                    "sent_id": DataType.int32(),
                    "sent_start": DataType.int32(),
                    "sent_end": DataType.int32(),
                    "sent_text": DataType.string(),
                    "sent_ents": DataType.list(DataType.string()),
                }
            )
        )
    )
    def chunk_text(self, text: list[str]):
        doc = self.nlp(text)
        return [
            {
                "sent_id": i,
                "sent_start": sent.start,
                "sent_end": sent.end,
                "sent_text": sent.text,
                "sent_ents": [ent.text for ent in sent.ents] if sent.ents else [],
            }
            for i, sent in enumerate(doc.sents)
        ]


if __name__ == "__main__":
    load_dotenv()

    PDF_URI = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
    TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
    GENERATION_MODEL = "gpt-5-nano"
    QUESTION = "What is Daft?"
    TOP_K = 3

    # Discover PDFs, extract their pages, and embed the text.
    documents = (
        daft.from_glob_path(PDF_URI)
        .with_column("pdf_file", file(col("path")))
        .with_column("pdf_pages", extract_pdf_pages(col("pdf_file")))
        .with_column(
            "page_embedding",
            embed_text(col("page_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
        )
    )

    # Encode the user query once.
    query = daft.from_pydict({"query_text": [QUESTION]}).with_column(
        "query_embedding",
        embed_text(col("query_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
    )

    # Rank all PDF chunks by cosine distance to the query embedding.
    ranked = (
        query.join(documents, how="cross")
        .with_column(
            "distance",
            cosine_distance(col("query_embedding"), col("page_embedding")),
        )
        .sort("distance")
    )

    top_matches = (
        ranked.select("query_text", "path", "page_number", "page_text", "distance")
        .limit(TOP_K)
        .collect()
        .to_pydict()
    )

    if not top_matches["path"]:
        raise RuntimeError("No PDF text was extracted — check your data source.")

    context_sections = []
    for path_value, page_number_value, page_text_value in zip(
        top_matches["path"],
        top_matches["page_number"],
        top_matches["page_text"],
    ):
        context_sections.append(
            f"Source: {path_value} (page {page_number_value})\n{page_text_value.strip()}"
        )

    context_blob = "\n\n---\n\n".join(context_sections)

    rag_df = daft.from_pydict(
        {
            "question": [QUESTION],
            "context": [context_blob],
        }
    ).with_column(
        "response",
        prompt(
            messages= [
                for x in y: 

                 
                    
                
            ],
            model=GENERATION_MODEL,
            provider="openai",
            reasoning={"effort": "medium"},
        ),
    )

    rag_df.select("question", "response").show(truncate=80)