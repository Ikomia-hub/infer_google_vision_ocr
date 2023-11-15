import copy
from ikomia import core, dataprocess
from google.cloud import vision
import os
import io
import cv2


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferGoogleVisionOcrParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.google_application_credentials = ''

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.window_size = int(params["window_size"])
        self.google_application_credentials = str(params["google_application_credentials"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {}
        params["google_application_credentials"] = str(self.google_application_credentials)
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferGoogleVisionOcr(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_output(dataprocess.CTextIO())
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferGoogleVisionOcrParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.client = None
        self.color = [255, 0, 0]

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Get input
        input = self.get_input(0)
        src_image = input.get_image()

        # Get output :
        text_output = self.get_output(1)
        output_dict = self.get_output(2)
        self.forward_input_image(0, 0)

        # Get parameters
        param = self.get_param_object()

        if self.client is None:
            if param.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = param.google_application_credentials
            self.client = vision.ImageAnnotatorClient()

        # Convert the NumPy array to a byte stream
        src_image = src_image[..., ::-1] # Convert to bgr
        is_success, image_buffer = cv2.imencode(".jpg", src_image)
        byte_stream = io.BytesIO(image_buffer)

        # Add confidence score's parameter
        ocr_detection_params = vision.TextDetectionParams(
                                        enable_text_detection_confidence_score=True
        )
        image_context = vision.ImageContext(text_detection_params=ocr_detection_params)

        # Infer
        response = self.client.text_detection(image=byte_stream, image_context=image_context)

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

        # Set output dict
        output_dict.data = {'Detected text': f'{response.text_annotations[0].description}'}

        # Get confidence scores from response
        scores = []
        scores = [word.confidence
                        for block in response.full_text_annotation.pages[0].blocks
                        for paragraph in block.paragraphs
                        for word in paragraph.words
        ]

        # Extract data and display output
        texts = response.text_annotations
        for i, text in enumerate(texts[1:]):
            word = f'{text.description}'
            conf = scores[i]

            # Get box coordinates
            vertices = [(vertex.x,vertex.y) for vertex in text.bounding_poly.vertices]

            # Extract all the x-coordinates and y-coordinates
            x_coords = [point[0] for point in vertices]
            y_coords = [point[1] for point in vertices]

            # Calculate x1, y1, w, and h
            x_box = min(x_coords)
            y_box = min(y_coords)
            w = max(x_coords) - x_box
            h = max(y_coords) - y_box

            # Add text graphics object
            text_output.add_text_field(
                            id=i,
                            label="",
                            text=word,
                            confidence=conf,
                            box_x=x_box,
                            box_y=y_box,
                            box_width=w,
                            box_height=h,
                            color=self.color
            )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferGoogleVisionOcrFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_google_vision_ocr"
        self.info.short_description = "Detects and extracts text from any image."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.icon_path = "images/cloud.png"
        self.info.path = "Plugins/Python/Text"
        self.info.version = "1.0.0"
        self.info.authors = "Google"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://cloud.google.com/vision/docs/ocr"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_google_vision_ocr"
        self.info.original_repository = "https://github.com/googleapis/python-vision"
        # Keywords used for search
        self.info.keywords = "Text detection,Text recognition,OCR,Cloud,Vision AI"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OCR"

    def create(self, param=None):
        # Create algorithm object
        return InferGoogleVisionOcr(self.info.name, param)
