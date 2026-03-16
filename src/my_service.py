from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import io
import json
from PIL import Image
from deepface import DeepFace
import numpy as np

settings = get_settings()

api_description = """
Analyse faces in images. Returns a JSON object with the following fields:
- age (Age of the person in the image),
- region (Region of the person in the image),
- gender (Gender of the person in the image),
- race (Race of the person in the image),
- dominant_race (Dominant race of the person in the image),
- emotion (Emotion of the person in the image),
- dominant_emotion (The dominant emotion of the person in the image)
"""

api_summary = """
Analyzes the faces in images.
"""

api_title = "Face Analyzer API."
version = "1.0.0"


class MyService(Service):
    """
    Face analyzer service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Face Analyzer",
            slug="face-analyzer",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/face-analyzer/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        # Get raw image data
        raw = data["image"].data
        buff = io.BytesIO(raw)
        img_pil = Image.open(buff)
        img = np.array(img_pil)
        diagnos = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=True,
            detector_backend="retinaface",
        )

        return {
            "result": TaskData(
                data=json.dumps(diagnos),
                type=FieldDescriptionType.APPLICATION_JSON
            )
        }
