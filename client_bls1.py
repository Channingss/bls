import os, sys
import numpy as np
import json
import tritongrpcclient
import argparse
import cv2
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        #default="ensemble_python_resnet50",
                        default="bls1",
                        help="Model name")
    parser.add_argument("--url",
                        type=str,
                        required=False,
                        default="localhost:8001",
                        help="Inference server URL. Default is localhost:8001.")
    parser.add_argument('-v',
                        "--verbose",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    args = parser.parse_args()

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    inputs = []
    outputs = []
    input_name = "INPUT_BLS1"
    output_name = "OUTPUT_BLS1"
    input = np.zeros([1]).astype(np.float32)

    inputs.append(
        tritongrpcclient.InferInput(input_name, input.shape, "FP32"))

    #params_data = np.array([[json.dumps(params)]], dtype=np.object_)
    #inputs.append(tritongrpcclient.InferInput('PARAMS', params_data.shape, "BYTES"))
    #inputs[1].set_data_from_numpy(params_data)
    outputs.append(tritongrpcclient.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(input)
    results = triton_client.infer(model_name=args.model_name,
                                  inputs=inputs,
                                  outputs=outputs)
    output = results.as_numpy(output_name)
    print(output)

    
    #maxs = np.argmax(output0_data, axis=1)
    #print(maxs)
    #print("Result is class: {}".format(labels_dict[maxs[0]]))
