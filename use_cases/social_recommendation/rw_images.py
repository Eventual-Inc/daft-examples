import os
import daft 
from daft import col, lit
from daft.functions import format, monotonically_increasing_id
from daft.io import IOConfig, S3Config
import numpy as np
from PIL import Image
import aioboto3

from daft.unity_catalog import UnityCatalog
from daft.catalog import Table
from dotenv import load_dotenv
load_dotenv()

@daft.cls()
class ImageWriter:
    def __init__(self):
        self.session = aioboto3.Session(
            region_name="us-west-2",
            aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("S3_SESSION_TOKEN")
        )

    async def write_images(self, image: np.ndarray, path: str) -> str:
        from io import BytesIO
        from urllib.parse import urlparse
        
        image = Image.fromarray(image)
        
        # Parse S3 path
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        # Write image to bytes buffer
        buffer = BytesIO()
        try:
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            async with self.session.client('s3') as s3_client:
                await s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            return path
        except Exception as e:
            raise ValueError(f"Error writing image to {path}: {e}")

if __name__ == "__main__":
    CATEGORY = os.getenv("CATEGORY", "general_visual_qna")

    DEST_URI = f"s3://daft-public-datasets/the_cauldron/all_images"
    UNITY_TABLE = "jaytest-unity.demo.all_images"


    # Configure UnityCatalog
    unity = UnityCatalog(
        endpoint=os.getenv("DATABRICKS_ENDPOINT"),
        token=os.getenv("DATABRICKS_TOKEN"),
    )

    daft.set_planning_config(
        default_io_config=IOConfig(
            s3=S3Config(
                region_name="us-west-2",
                key_id=os.getenv("S3_ACCESS_KEY_ID"),
                access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                session_token=os.getenv("S3_SESSION_TOKEN"),
            )
        )
    )

    deltalake_config = {
        "checkpoint.writeStatsAsJson": "false",
        "checkpoint.writeStatsAsStruct": "true",
        "enableDeletionVectors": "true",
        "feature.appendOnly": "supported",
        "feature.deletionVectors": "supported",
        "feature.invariants": "supported",
        "minReaderVersion": "3",
        "minWriterVersion": "7",
        "parquet.compression.codec": "zstd"
    }

    image_writer = ImageWriter()
    #print(unity.list_tables('jaytest-unity.demo'))
    #unity_table = unity.load_table(UNITY_TABLE)
    #unity_table.table_info()

    
    for SUBSET in ["vqav2", "tallyqa", "aokvqa"]:
        SOURCE_URI = f"s3://daft-public-datasets/the_cauldron/original/{CATEGORY}/{SUBSET}/*.parquet"

        df = daft.read_parquet(SOURCE_URI)
        print(df.count_rows())
        df = df.with_column("id", monotonically_increasing_id()).where(col("id") > 29524)
        df = df.with_column("subset", lit(SUBSET))
        df = df.with_column("category", lit(CATEGORY))
        df = df.with_column("image_hash", col("image").encode_image("png").hash())
        df = df.with_column("image_path", format("{}/{}_{}_xxhash{}.png", lit(DEST_URI), lit(CATEGORY), lit(SUBSET), col("image_hash")))
        df = df.with_column("image_written", image_writer.write_images(col("image"), col("image_path")))
        
        
        df.select(col("image_written").alias("uri"), col("subset"), col("category"), col("image_hash")).write_deltalake(f"{DEST_URI}/_index.deltalake", mode="overwrite")
    

