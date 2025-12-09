import os
import daft
from daft.io import S3Config, IOConfig
from dotenv import load_dotenv

load_dotenv()

S3_URI = "s3://daft-public-datasets/the_cauldron/original/"

subsets = {
    "general_visual_qna": [
        "vqav2",
        "cocoqa",
        "visual7w",
        "aokvqa",
        "tallyqa",
        "okvqa",
        "hateful_memes",
        "vqarad",
    ],
    "captioning": ["localized_narratives", "screen2words", "vsr"],
    "ocr_doc_understanding": [
        "rendered_text",
        "docvqa",
        "textcaps",
        "textvqa",
        "st_vqa",
        "ocrvqa",
        "visualmrc",
        "iam",
        "infographic_vqa",
        "diagram_image_to_text",
    ],
    "chart_figure_understanding": [
        "chart2text",
        "dvqa",
        "vistext",
        "chartqa",
        "plotqa",
        "figureqa",
        "mapqa",
    ],
    "table_understanding": [
        "tabmwp",
        "tat_qa",
        "hitab",
        "multihiertt",
        "finqa",
        "robut_wikisql",
        "robut_sqa",
        "robut_wtq",
    ],
    "reasoning_logic_maths": [
        "geomverse",
        "clevr_math",
        "clevr",
        "iconqa",
        "raven",
        "intergps",
    ],
    "textbook_academic_questions": ["ai2d", "tqa", "scienceqa"],
    "differences_between_2_images": ["nlvr2", "mimic_cgd", "spot_the_diff"],
    "screenshot_to_code": ["websight", "datikz"],
}

CATEGORY = os.getenv("CATEGORY", None)
if CATEGORY is None:
    raise ValueError("CATEGORY environment variable is not set")

if CATEGORY in ["captioning", "general_visual_qna"]:
    raise ValueError(
        f"You already ran {CATEGORY}! Dont process 100's of GBs of data again."
    )

WRITE_MODE = os.getenv("WRITE_MODE", "overwrite")

for subset in subsets[CATEGORY]:
    DEST_URI = S3_URI + CATEGORY + "/" + subset

    df = daft.read_parquet(
        f"hf://datasets/HuggingFaceM4/the_cauldron/{subset}/*.parquet"
    )
    df = df.explode("images").with_column(
        "image", daft.col("images")["bytes"].decode_image().convert_image("RGB")
    )
    df.select("image", "texts").write_parquet(
        DEST_URI,
        write_mode=WRITE_MODE,
        io_config=IOConfig(
            s3=S3Config(
                region_name="us-west-2",
                key_id=os.getenv("S3_ACCESS_KEY_ID"),
                access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                session_token=os.getenv("S3_SESSION_TOKEN"),
            )
        ),
    )
