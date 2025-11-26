import os
import daft
from daft.functions import prompt

SUBSET = os.environ.get("SUBSET", "ai2d")
LIMIT = os.environ.get("LIMIT", 100)
DEST_URI = os.environ.get("DEST_URI", None)

def run_prompt_text(df: daft.DataFrame): 
    df = (
        df
        .with_column(
            "response",
            prompt(
                [daft.col("image"), daft.col("texts")["user"]],
                model="Qwen/Qwen3-VL-4B-Instruct",
                provider="daft",
            ),
        )
        .limit(LIMIT)
    )
    return df

def main():
    df = daft.read_parquet(f"hf://datasets/HuggingFaceM4/the_cauldron/{SUBSET}/*.parquet")
    df = df.explode("images").with_column("image", daft.col("images")["bytes"].decode_image())
    df = df.explode("texts")
    
    df = run_prompt_text(df)
    
    if DEST_URI is not None:
        df.write_parquet(DEST_URI)
    else:
        df.show()
    

if __name__ == "__main__":
    main()
