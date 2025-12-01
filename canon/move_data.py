import os
import daft 
from daft.io import S3Config, IOConfig
#from dotenv import load_dotenv

#load_dotenv()

S3_URI = "s3://daft-public-datasets/the_cauldron/original/"

subsets = {
    "general visual question answering": ["vqav2", "coco-qa", "visual7w", "a-okvqa", "tallyqa", "ok-vqa", "hatefulmemes", "vqa-rad"],
    "captioning": ["lnarratives", "screen2words", "vsr"],
    "ocr, document understanding, text transcription": ["renderedtext", "docvqa", "textcaps", "textvqa", "st-vqa", "ocr-vqa", "visualmrc", "iam", "infovqa", "diagram image-to-text"],
    "chart/figure understanding": ["chart2text", "dvqa", "vistext", "chartqa", "plotqa", "figureqa", "mapqa"],
    "table understanding": ["tabmwp", "tat-qa", "hitab", "multihiertt", "finqa", "wikisql", "sqa", "wtq"],
    "reasoning, logic, maths": ["geomverse", "clevr-math", "clevr", "iconqa", "raven", "inter-gps"],
    "textbook/academic questions": ["ai2d", "tqa", "scienceqa"],
    "differences between 2 images": ["nlvr2", "gsd", "spot the diff"],
    "screenshot to code": ["websight", "datikz"],
}

for category in subsets.keys():
    for subset in subsets[category]:
        DEST_URI = S3_URI + category + "/" + subset

        df = daft.read_parquet(f"hf://datasets/HuggingFaceM4/the_cauldron/{subset}/*.parquet")
        df = df.explode("images").with_column("image", daft.col("images")["bytes"].decode_image().convert_image("RGB"))
        df.select("image", "texts").limit(10).write_parquet(
            DEST_URI,
            io_config=IOConfig(
                s3 = S3Config(
                    region_name="us-west-2",
                    key_id=os.getenv("S3_ACCESS_KEY_ID"),
                    access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                    session_token=os.getenv("S3_SESSION_TOKEN"),
                )
            )
        )
