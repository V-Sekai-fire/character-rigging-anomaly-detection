# anomalib train --config avatar_rigging.yaml # --print_config

from cog import File
import io

def train(param: str) -> File:
    return io.StringIO("hello " + param)