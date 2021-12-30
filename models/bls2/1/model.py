import os
import sys
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input_bls_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_BLS2")
            out = np.zeros([0]).astype(np.float32)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUTPUT_BLS2", out)])
            responses.append(inference_response)
        return responses
