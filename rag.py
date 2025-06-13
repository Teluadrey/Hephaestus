from typing import List
import os
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper


converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction
# --------------------------------------------------------------
directory = "../Hephaestus/docs"

documents = []
for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        filepath = os.path.join(directory, filename)
        documents.append(filepath)

conv_results_iter = converter.convert_all(documents)
docs = []
for result in conv_results_iter:
    if result.document:
        document = result.document
        docs.append(document)

# --------------------------------------------------------------
markdowns = []
for doc in docs:
    markdown_output = doc.export_to_markdown()
    markdowns.append(markdown_output)

jsons = []
for doc in docs:
    json_output = doc.export_to_dict()
    jsons.append(json_output)


# --------------------------------------------------------------
# Chunking
# --------------------------------------------------------------
load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()

tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)
chunks = []
for doc in docs:
    chunk_iter = chunker.chunk(doc)
    chunks.extend(list(chunk_iter))


# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------

# Create a LanceDB database
db = lancedb.connect("data/lancedb")


# Get the OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table("hpts", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create table with processed chunks
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------

table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table.to_pandas()
table.count_rows()


# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table("hpts")


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------

result = table.search(
    query="what are some key safety considirations while creating design for nuclear reactor",
    query_type="vector",
).limit(3)
result.to_pandas()
