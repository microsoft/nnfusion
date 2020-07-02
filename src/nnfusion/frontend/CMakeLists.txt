# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if (NNFUSION_TENSORFLOW_IMPORT_ENABLE)
    add_subdirectory(tensorflow_import)
    if (NGRAPH_ONNX_IMPORT_ENABLE)
        add_subdirectory(onnx_import)
    endif()
endif()