import os
import daft 
from daft.io import S3Config, IOConfig
#from dotenv import load_dotenv

#load_dotenv()

S3_URI = "s3://daft-public-datasets/the_cauldron/original/"

subsets = {
    "general visual question answering": ["vqav2", "cocoqa", "visual7w", "aokvqa", "tallyqa", "okvqa", "hateful_memes", "vqarad"],
    "captioning": ["localized_narratives", "screen2words", "vsr"],
    "ocr, document understanding, text transcription": ["rendered_text", "docvqa", "textcaps", "textvqa", "st_vqa", "ocrvqa", "visualmrc", "iam", "infographic_vqa", "diagram_image_to_text"],
    "chart/figure understanding": ["chart2text", "dvqa", "vistext", "chartqa", "plotqa", "figureqa", "mapqa"],
    "table understanding": ["tabmwp", "tat_qa", "hitab", "multihiertt", "finqa", "robut_wikisql", "robut_sqa", "robut_wtq"],
    "reasoning, logic, maths": ["geomverse", "clevr_math", "clevr", "iconqa", "raven", "intergps"],
    "textbook/academic questions": ["ai2d", "tqa", "scienceqa"],
    "differences between 2 images": ["nlvr2", "mimic_cgd", "spot_the_diff"],
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
