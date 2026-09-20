from pertbench_long.schemas.jsonschema_export import write_json_schemas
from pathlib import Path
write_json_schemas(Path(__file__).parent / "jsonschema")
