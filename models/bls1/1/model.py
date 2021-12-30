import os
import sys
import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            input_bls_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_BLS1")
            inputs = pb_utils.Tensor("INPUT_BLS2",  input_bls_tensor.as_numpy())
            inference_request = pb_utils.InferenceRequest(
              model_name='bls2',
              requested_output_names=['OUTPUT_BLS2'],
              inputs=[inputs])
            inference_response = inference_request.exec()
            output_bls2_tensor = pb_utils.get_output_tensor_by_name(inference_response, 'OUTPUT_BLS2')
            output_bls2 =  from_dlpack(output_bls2_tensor.to_dlpack()).numpy()
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("OUTPUT_BLS1", output_bls2)])
            responses.append(inference_response)
        return responses
